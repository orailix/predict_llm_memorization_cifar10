import argparse
import json
import os
import pathlib
import typing
import warnings

import dotenv
import filelock
import mlflow
import numpy as np
import scipy.optimize
import sklearn.metrics
import torch
import torch.utils.data
import torchdata.dataloader2
import torchdata.datapipes.map
import torchvision
import torchvision.transforms.v2
import tqdm
from loguru import logger

import attack_util
import base
import data
import models

NUM_TRAIN_EPOCHS = 200


def main():
    dotenv.load_dotenv()
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    experiment_base_dir = args.experiment_dir.expanduser().resolve()

    experiment_name = args.experiment
    run_suffix = args.run_suffix
    verbose = args.verbose

    global_seed = args.seed
    base.setup_seeds(global_seed)

    num_shadow = args.num_shadow
    assert num_shadow > 0
    num_canaries = args.num_canaries
    assert num_canaries > 0
    num_poison = args.num_poison
    assert num_poison >= 0

    canary_type = data.CanaryType(args.canary_type)

    data_generator = data.DatasetGenerator(
        num_shadow=num_shadow,
        num_canaries=num_canaries,
        canary_type=canary_type,
        num_poison=num_poison,
        poison_type=data.PoisonType(args.poison_type),
        data_dir=data_dir,
        seed=global_seed,
        download=bool(os.environ.get("DOWNLOAD_DATA")),
    )
    directory_manager = DirectoryManager(
        experiment_base_dir=experiment_base_dir,
        experiment_name=experiment_name,
        run_suffix=run_suffix,
    )

    num_teachers = args.num_teachers
    num_query = args.num_query
    assert 0 < num_query < num_teachers

    if args.action == "attack":
        # Attack only depends on global seed (if any)
        _run_attack(args, canary_type, data_generator, directory_manager)
    elif args.action == "train":
        shadow_model_idx = args.shadow_model_idx
        assert 0 <= shadow_model_idx < num_shadow
        # Setting seed is the same for all teachers in a fixed shadow model to ensure
        #  training data splits are consistent.
        setting_seed = base.get_setting_seed(
            global_seed=global_seed,
            shadow_model_idx=shadow_model_idx,
            num_shadow=num_shadow,
        )
        base.setup_seeds(setting_seed)

        (
            full_data,
            membership_mask,
            canary_mask,
            poison_mask,
        ) = data_generator.build_train_data_full_with_poison(shadow_model_idx)

        splitai_handler = SplitAIHandler(
            setting_seed=setting_seed,
            num_teachers=num_teachers,
            num_query=num_query,
            full_data=full_data,
            membership_mask=membership_mask,
            canary_mask=canary_mask,
            poison_mask=poison_mask,
        )

        train_action = args.train_action
        if train_action == "splitai":
            teacher_idx = args.teacher_idx
            assert 0 <= teacher_idx < num_teachers

            # Make sure each teacher has a different training seed, but be reproducible
            teacher_training_seeds = np.random.default_rng(seed=setting_seed).integers(
                0,
                2**32,
                size=(num_teachers,),
                dtype=np.uint32,
            )
            _run_train_splitai(
                args,
                teacher_idx,
                shadow_model_idx,
                data_generator,
                splitai_handler,
                directory_manager,
                teacher_training_seeds[teacher_idx],
                experiment_name,
                run_suffix,
                verbose,
            )
        elif train_action == "distillation":
            _run_train_distillation(
                args,
                shadow_model_idx,
                data_generator,
                splitai_handler,
                directory_manager,
                setting_seed,
                experiment_name,
                run_suffix,
                verbose,
            )
        else:
            assert False, f"Unknown training action {train_action}"
    else:
        assert False, f"Unknown action {args.action}"


def _run_train_splitai(
    args: argparse.Namespace,
    teacher_idx: int,
    shadow_model_idx: int,
    data_generator: data.DatasetGenerator,
    splitai_handler: "SplitAIHandler",
    directory_manager: "DirectoryManager",
    training_seed: int,
    experiment_name: str,
    run_suffix: typing.Optional[str],
    verbose: bool,
) -> None:
    logger.info(f"Training teacher {teacher_idx} for shadow model {shadow_model_idx}")
    logger.info(
        f"{data_generator.num_canaries} canaries ({data_generator.canary_type.value}), "
        f"{data_generator.num_poison} poisons ({data_generator.poison_type.value})"
    )

    output_dir = directory_manager.get_training_teacher_output_dir(
        shadow_model_idx, teacher_idx
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = directory_manager.get_training_log_dir(shadow_model_idx, teacher_idx)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_data = splitai_handler.build_teacher_train_data(teacher_idx=teacher_idx)

    # Make sure only one run creates the MLFlow experiment and starts at a time to avoid concurrency issues
    with filelock.FileLock(log_dir / "enter_mlflow.lock"):
        mlflow.set_tracking_uri(f"file:{log_dir}")
        mlflow.set_experiment(experiment_name=experiment_name)
        run_name = f"teacher_{shadow_model_idx}_{teacher_idx}"
        if run_suffix is not None:
            run_name += f"_{run_suffix}"
        run = mlflow.start_run(run_name=run_name)
    with run:
        mlflow.log_params(
            {
                "shadow_model_idx": shadow_model_idx,
                "num_canaries": data_generator.num_canaries,
                "canary_type": data_generator.canary_type.value,
                "num_poison": data_generator.num_poison,
                "poison_type": data_generator.poison_type.value,
                "training_seed": training_seed,
                "teacher_idx": teacher_idx,
            }
        )
        current_model = _train_model(
            train_data,
            training_seed=training_seed,
            verbose=verbose,
        )
        current_model.eval()

        torch.save(current_model, output_dir / "model.pt")
        logger.info("Saved model")

        metrics = dict()

        # Always predict on full training data, including duplicates/poisons, even if a bit redundant
        # Data augmentation with Split-AI is ill-defined and not needed, hence not done here
        logger.info("Predicting logits and evaluating full training data")
        full_train_data = splitai_handler.get_full_train_data()
        train_data_full_pipe = full_train_data.as_unlabeled().build_datapipe()
        train_pred_full = _predict(
            current_model, train_data_full_pipe, data_augmentation=False
        )
        torch.save(train_pred_full, output_dir / "predictions_train.pt")

        # NB: Training IN refers to teacher membership, not full SELENA membership
        train_membership_mask, canary_mask = splitai_handler.get_teacher_masks(
            teacher_idx
        )
        train_ys_pred = torch.argmax(train_pred_full[:, 0], dim=-1)
        train_ys = full_train_data.targets
        correct_predictions_train = torch.eq(train_ys_pred, train_ys).to(
            dtype=base.DTYPE_EVAL
        )
        metrics.update(
            {
                "train_accuracy_full": torch.mean(correct_predictions_train).item(),
                "train_accuracy_in": torch.mean(
                    correct_predictions_train[train_membership_mask]
                ).item(),
                "train_accuracy_out": torch.mean(
                    correct_predictions_train[~train_membership_mask]
                ).item(),
            }
        )
        logger.info(f"Train accuracy (full data): {metrics['train_accuracy_full']:.4f}")
        logger.info(
            f"Train accuracy (only IN samples): {metrics['train_accuracy_in']:.4f}"
        )
        logger.info(
            f"Train accuracy (only OUT samples): {metrics['train_accuracy_out']:.4f}"
        )
        metrics.update(
            {
                "train_accuracy_canaries": torch.mean(
                    correct_predictions_train[canary_mask]
                ).item(),
                "train_accuracy_canaries_in": torch.mean(
                    correct_predictions_train[canary_mask & train_membership_mask]
                ).item(),
                "train_accuracy_canaries_out": torch.mean(
                    correct_predictions_train[canary_mask & (~train_membership_mask)]
                ).item(),
            }
        )
        logger.info(
            f"Train accuracy (full canary subset): {metrics['train_accuracy_canaries']:.4f}"
        )
        logger.info(
            f"Train accuracy (IN canary subset): {metrics['train_accuracy_canaries_in']:.4f}"
        )
        logger.info(
            f"Train accuracy (OUT canary subset): {metrics['train_accuracy_canaries_out']:.4f}"
        )

        logger.info("Evaluating on test data")
        test_metrics, test_pred = _evaluate_model_test(current_model, data_generator)
        metrics.update(test_metrics)
        logger.info(f"Test accuracy: {metrics['test_accuracy']:.4f}")
        torch.save(test_pred, output_dir / "predictions_test.pt")
        mlflow.log_metrics(metrics, step=NUM_TRAIN_EPOCHS)
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)


def _run_train_distillation(
    args: argparse.Namespace,
    shadow_model_idx: int,
    data_generator: data.DatasetGenerator,
    splitai_handler: "SplitAIHandler",
    directory_manager: "DirectoryManager",
    training_seed: int,
    experiment_name: str,
    run_suffix: typing.Optional[str],
    verbose: bool,
) -> None:
    logger.info(f"Distilling student for shadow model {shadow_model_idx}")
    logger.info(
        f"{data_generator.num_canaries} canaries ({data_generator.canary_type.value}), "
        f"{data_generator.num_poison} poisons ({data_generator.poison_type.value})"
    )

    output_dir = directory_manager.get_training_student_output_dir(shadow_model_idx)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = directory_manager.get_training_log_dir(shadow_model_idx, teacher_idx=None)
    log_dir.mkdir(parents=True, exist_ok=True)

    test_data = data_generator.build_test_data()
    test_ys = test_data.targets

    logger.info("Loading raw teacher predictions")
    teacher_predictions_test = []
    teacher_predictions_train = []
    for teacher_idx in range(splitai_handler.num_teachers):
        # Remove data augmentation dimensions while loading
        teacher_dir = directory_manager.get_training_teacher_output_dir(
            shadow_model_idx, teacher_idx
        )
        teacher_predictions_test.append(
            torch.load(teacher_dir / "predictions_test.pt")[:, 0]
        )
        teacher_predictions_train.append(
            torch.load(teacher_dir / "predictions_train.pt")[:, 0]
        )
    teacher_predictions_test = torch.stack(teacher_predictions_test, dim=1)
    teacher_predictions_train = torch.stack(teacher_predictions_train, dim=1)
    assert teacher_predictions_train.dim() == 3
    assert teacher_predictions_train.size()[1:] == teacher_predictions_test.size()[1:]

    # Make sure only one run creates the MLFlow experiment and starts at a time to avoid concurrency issues
    with filelock.FileLock(log_dir / "enter_mlflow.lock"):
        mlflow.set_tracking_uri(f"file:{log_dir}")
        mlflow.set_experiment(experiment_name=experiment_name)
        run_name = f"student_{shadow_model_idx}"
        if run_suffix is not None:
            run_name += f"_{run_suffix}"
        run = mlflow.start_run(run_name=run_name)
    with run:
        mlflow.log_params(
            {
                "shadow_model_idx": shadow_model_idx,
                "num_canaries": data_generator.num_canaries,
                "canary_type": data_generator.canary_type.value,
                "num_poison": data_generator.num_poison,
                "poison_type": data_generator.poison_type.value,
                "training_seed": training_seed,
                "teacher_idx": None,
            }
        )

        metrics = dict()

        # Evaluate ensemble and split-ai, and infer training soft-labels
        logger.info("Performing Split-AI inference")
        metrics.update(
            splitai_handler.evaluate_ensemble(teacher_predictions_test, test_ys)
        )
        metrics.update(
            splitai_handler.evaluate_splitai_test(teacher_predictions_test, test_ys)
        )
        del teacher_predictions_test
        splitai_unreduced_logits = splitai_handler.infer_splitai_train_full(
            teacher_predictions_train
        )
        torch.save(
            splitai_unreduced_logits, output_dir / "splitai_predictions_train_raw.pt"
        )
        splitai_probs_train = splitai_handler.softlabels_from_raw(
            splitai_unreduced_logits
        )
        train_data = splitai_handler.build_distillation_dataset(splitai_probs_train)
        (
            splitai_membership_mask,
            splitai_train_ys,
            splitai_canary_mask,
            splitai_poison_mask,
        ) = splitai_handler.get_splitai_eval_data()

        # Evaluate Split-AI
        logger.info("Evaluating Split-AI")
        logger.info(f"Split-AI test accuracy: {metrics['splitai_test_accuracy']:.4f}")

        splitai_train_ys_pred = torch.argmax(splitai_probs_train, dim=-1)
        splitai_correct_train = torch.eq(splitai_train_ys_pred, splitai_train_ys).to(
            dtype=base.DTYPE_EVAL
        )
        metrics.update(
            {
                "splitai_train_accuracy_full": torch.mean(splitai_correct_train).item(),
                "splitai_train_accuracy_in": torch.mean(
                    splitai_correct_train[splitai_membership_mask]
                ).item(),
                "splitai_train_accuracy_out": torch.mean(
                    splitai_correct_train[~splitai_membership_mask]
                ).item(),
            }
        )
        logger.info(
            f"Split-AI train accuracy (full data w/ duplicates): {metrics['splitai_train_accuracy_full']:.4f}"
        )
        logger.info(
            f"Split-AI train accuracy (only IN samples w/ duplicates): {metrics['splitai_train_accuracy_in']:.4f}"
        )
        logger.info(
            f"Split-AI train accuracy (only OUT samples): {metrics['splitai_train_accuracy_out']:.4f}"
        )
        metrics.update(
            {
                "splitai_train_accuracy_canaries": torch.mean(
                    splitai_correct_train[splitai_canary_mask]
                ).item(),
                "splitai_train_accuracy_canaries_in": torch.mean(
                    splitai_correct_train[splitai_canary_mask & splitai_membership_mask]
                ).item(),
                "splitai_train_accuracy_canaries_out": torch.mean(
                    splitai_correct_train[
                        splitai_canary_mask & (~splitai_membership_mask)
                    ]
                ).item(),
            }
        )
        logger.info(
            f"Split-AI train accuracy (full canary subset): {metrics['splitai_train_accuracy_canaries']:.4f}"
        )
        logger.info(
            f"Split-AI train accuracy (IN canary subset): {metrics['splitai_train_accuracy_canaries_in']:.4f}"
        )
        logger.info(
            f"Split-AI train accuracy (OUT canary subset): {metrics['splitai_train_accuracy_canaries_out']:.4f}"
        )

        current_model = _train_model(
            train_data,
            training_seed=training_seed,
            verbose=verbose,
        )
        current_model.eval()

        torch.save(current_model, output_dir / "model.pt")
        logger.info("Saved model")

        logger.info("Predicting logits and evaluating full training data")
        full_train_data = data_generator.build_full_train_data()
        train_data_full_pipe = full_train_data.as_unlabeled().build_datapipe()
        # NB: Always predict on augmented samples, even if not training with data augmentation
        train_pred_full = _predict(
            current_model, train_data_full_pipe, data_augmentation=True
        )
        torch.save(train_pred_full, output_dir / "predictions_train.pt")

        train_membership_mask = data_generator.build_in_mask(
            shadow_model_idx
        )  # does not include poisons
        train_ys_pred = torch.argmax(train_pred_full[:, 0], dim=-1)
        train_ys = full_train_data.targets
        correct_predictions_train = torch.eq(train_ys_pred, train_ys).to(
            dtype=base.DTYPE_EVAL
        )
        metrics.update(
            {
                "train_accuracy_full": torch.mean(correct_predictions_train).item(),
                "train_accuracy_in": torch.mean(
                    correct_predictions_train[train_membership_mask]
                ).item(),
                "train_accuracy_out": torch.mean(
                    correct_predictions_train[~train_membership_mask]
                ).item(),
            }
        )
        logger.info(f"Train accuracy (full data): {metrics['train_accuracy_full']:.4f}")
        logger.info(
            f"Train accuracy (only IN samples): {metrics['train_accuracy_in']:.4f}"
        )
        logger.info(
            f"Train accuracy (only OUT samples): {metrics['train_accuracy_out']:.4f}"
        )
        canary_mask = torch.zeros_like(train_membership_mask)
        canary_mask[data_generator.get_canary_indices()] = True
        metrics.update(
            {
                "train_accuracy_canaries": torch.mean(
                    correct_predictions_train[canary_mask]
                ).item(),
                "train_accuracy_canaries_in": torch.mean(
                    correct_predictions_train[canary_mask & train_membership_mask]
                ).item(),
                "train_accuracy_canaries_out": torch.mean(
                    correct_predictions_train[canary_mask & (~train_membership_mask)]
                ).item(),
            }
        )
        logger.info(
            f"Train accuracy (full canary subset): {metrics['train_accuracy_canaries']:.4f}"
        )
        logger.info(
            f"Train accuracy (IN canary subset): {metrics['train_accuracy_canaries_in']:.4f}"
        )
        logger.info(
            f"Train accuracy (OUT canary subset): {metrics['train_accuracy_canaries_out']:.4f}"
        )

        logger.info("Evaluating on test data")
        test_metrics, test_pred = _evaluate_model_test(current_model, data_generator)
        metrics.update(test_metrics)
        logger.info(f"Test accuracy: {metrics['test_accuracy']:.4f}")
        torch.save(test_pred, output_dir / "predictions_test.pt")
        mlflow.log_metrics(metrics, step=NUM_TRAIN_EPOCHS)
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)


def _evaluate_model_test(
    model: torch.nn.Module,
    data_generator: data.DatasetGenerator,
    disable_tqdm: bool = False,
) -> typing.Tuple[typing.Dict[str, float], torch.Tensor]:
    test_data = data_generator.build_test_data()
    test_ys = test_data.targets
    test_xs_datapipe = test_data.as_unlabeled().build_datapipe()
    test_pred = _predict(
        model, test_xs_datapipe, data_augmentation=False, disable_tqdm=disable_tqdm
    )
    test_ys_pred = torch.argmax(test_pred[:, 0], dim=-1)
    correct_predictions = torch.eq(test_ys_pred, test_ys).to(base.DTYPE_EVAL)
    return {
        "test_accuracy": torch.mean(correct_predictions).item(),
    }, test_pred


def _run_attack(
    args: argparse.Namespace,
    canary_type: data.CanaryType,
    data_generator: data.DatasetGenerator,
    directory_manager: "DirectoryManager",
) -> None:
    output_dir = directory_manager.get_attack_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        attack_ys,
        shadow_membership_mask,
        canary_indices,
    ) = data_generator.build_attack_data()
    labels_file = output_dir / "attack_ys.pt"
    torch.save(attack_ys, labels_file)
    logger.info(f"Saved audit ys to {labels_file}")
    canary_indices_file = output_dir / "canary_indices.pt"
    torch.save(canary_indices, canary_indices_file)
    logger.info(f"Saved canary indices to {canary_indices_file}")

    assert shadow_membership_mask.size() == (
        data_generator.num_raw_training_samples,
        data_generator.num_shadow,
    )
    membership_file = output_dir / "shadow_membership_mask.pt"
    torch.save(shadow_membership_mask, membership_file)
    logger.info(f"Saved membership to {membership_file}")

    # For DUPLICATES_MISLABEL_HALF:
    # - canary_indices are w.r.t. the mislabeled samples (which we need to query for student)
    # - canary_mask_sai are the corresponding CLEAN samples (which we need to query for Split-AI!)
    # - attack_ys, attack_ys_sai is for all samples
    # - shadow_membership_mask, shadow_membership_mask_sai, poison_mask_sai is for all samples
    # => Want membership and labels w.r.t. mislabeled samples, but query Split-AI on clean samples

    # Load attack data (with poison!)
    (
        attack_ys_sai,
        shadow_membership_mask_sai,
        canary_mask_sai,
        poison_mask_sai,
    ) = data_generator.build_attack_data_with_poison()

    # Load Split-AI logits
    shadow_splitai_logits_raw = []
    for shadow_model_idx in range(data_generator.num_shadow):
        shadow_model_dir = directory_manager.get_training_student_output_dir(
            shadow_model_idx
        )
        shadow_splitai_logits_raw.append(
            torch.load(shadow_model_dir / "splitai_predictions_train_raw.pt")
        )
    shadow_splitai_logits = torch.stack(shadow_splitai_logits_raw, dim=1).to(
        dtype=base.DTYPE_EVAL
    )
    # FIXME: Could do more numerically stable logit score, and also hinge now, since we have actual non-averaged logits
    #  but does not seem to matter
    splitai_scores_full = {
        "logit": attack_util.logit_score_from_probs(
            SplitAIHandler.softlabels_from_raw(shadow_splitai_logits), attack_ys_sai
        ),
    }
    splitai_scores_full = {
        score_name: scores.unsqueeze(2)
        for score_name, scores in splitai_scores_full.items()
    }

    # Attack Split-AI, query on canaries
    logger.info("# Split-AI, query on canaries")
    _run_individual_attack(
        {
            score_name: scores[canary_mask_sai]
            for score_name, scores in splitai_scores_full.items()
        },
        shadow_membership_mask_sai[canary_mask_sai],
        output_dir,
        prefix="splitai_canaries",
    )

    # Attack Split-AI, query on duplicates (/poison)
    if data_generator.num_poison > 0:
        assert data_generator.num_canaries == data_generator.num_poison
        logger.info("# Split-AI, query on poison")
        _run_individual_attack(
            {
                score_name: scores[poison_mask_sai]
                for score_name, scores in splitai_scores_full.items()
            },
            shadow_membership_mask_sai[canary_indices],
            output_dir,
            prefix="splitai_poison",
        )

    # Attack Split-AI, query on near-duplicates
    if (
        args.embeddings_file is not None
        and canary_type != data.CanaryType.DUPLICATES_MISLABEL_HALF
    ):
        # Indices of non-canaries non-poison samples
        other_indices = torch.where((~canary_mask_sai) & (~poison_mask_sai))[0]
        assert other_indices.size() == (
            data_generator.num_raw_training_samples - data_generator.num_canaries,
        )

        embeddings = torch.load(args.embeddings_file)
        other_embeddings = embeddings[other_indices]
        other_embeddings = other_embeddings / other_embeddings.norm(dim=1, keepdim=True)
        canary_embeddings = embeddings[canary_indices]  # Use canary order!
        canary_embeddings = canary_embeddings / canary_embeddings.norm(
            dim=1, keepdim=True
        )

        # Compute similarities, (num canaries, num non-canaries non-poison)
        similarities = canary_embeddings @ other_embeddings.T

        # Determine maximum weight matching to resolve potential same nearest neighbor for multiple canaries
        canary_nn_indices, other_nn_indices = scipy.optimize.linear_sum_assignment(
            similarities, maximize=True
        )
        assert np.all(canary_nn_indices == np.arange(len(canary_indices)))

        # Map NN indices w.r.t. embedding back to indices w.r.t. full dataset
        nearest_neighbor_indices_emb = torch.from_numpy(other_nn_indices)
        nearest_neighbor_indices = other_indices[nearest_neighbor_indices_emb]

        splitai_scores = {
            # Need to calculate score on nearest neighbors w.r.t. canary labels,
            #  because those labels are leaking when querying on near-duplicates
            "logit": attack_util.logit_score_from_probs(
                SplitAIHandler.softlabels_from_raw(
                    shadow_splitai_logits[nearest_neighbor_indices]
                ),
                attack_ys_sai[canary_indices],
            ).unsqueeze(2),
        }

        logger.info("# Split-AI, query on near-duplicates")
        _run_individual_attack(
            splitai_scores,
            shadow_membership_mask_sai[canary_indices],
            output_dir,
            prefix="splitai_nn",
        )

    # Attack distilled student

    # Load logits
    shadow_logits_raw = []
    for shadow_model_idx in range(data_generator.num_shadow):
        shadow_model_dir = directory_manager.get_training_student_output_dir(
            shadow_model_idx
        )
        shadow_logits_raw.append(torch.load(shadow_model_dir / "predictions_train.pt"))
    shadow_logits = torch.stack(shadow_logits_raw, dim=1)
    assert shadow_logits.dim() == 4  # samples x shadow models x augmentations x classes
    assert shadow_logits.size(0) == data_generator.num_raw_training_samples
    assert shadow_logits.size(1) == data_generator.num_shadow
    num_augmentations = shadow_logits.size(2)

    student_scores = {
        "hinge": attack_util.hinge_score(
            shadow_logits[canary_indices], attack_ys[canary_indices]
        ),
        "logit": attack_util.logit_score(
            shadow_logits[canary_indices], attack_ys[canary_indices]
        ),
    }

    if canary_type != data.CanaryType.DUPLICATES_MISLABEL_HALF:
        assert all(
            scores.size()
            == (
                data_generator.num_canaries,
                data_generator.num_shadow,
                num_augmentations,
            )
            for scores in student_scores.values()
        )
    else:
        assert all(
            scores.size()
            == (
                data_generator.num_canaries // 2,
                data_generator.num_shadow,
                num_augmentations,
            )
            for scores in student_scores.values()
        )
    logger.info("# Distilled student")
    _run_individual_attack(
        student_scores,
        shadow_membership_mask[canary_indices],
        output_dir,
        prefix="student",
    )


def _run_individual_attack(
    shadow_scores: typing.Dict[str, torch.Tensor],
    shadow_membership_mask: torch.Tensor,
    output_dir: pathlib.Path,
    prefix: str,
) -> None:
    assert all(scores.dim() == 3 for scores in shadow_scores.values())
    assert all(
        scores.size(0) == shadow_membership_mask.size(0)
        for scores in shadow_scores.values()
    )  # num samples
    assert all(
        scores.size(1) == shadow_membership_mask.size(1)
        for scores in shadow_scores.values()
    )  # num shadow
    num_augmentations = next(iter(shadow_scores.values())).size(2)
    assert all(scores.size(2) == num_augmentations for scores in shadow_scores.values())

    # Global threshold
    logger.info("## Global threshold")
    for score_name, scores in shadow_scores.items():
        logger.info(f"### {score_name}")
        # Use score on first data augmentation (= no augmentations)
        # => scores and membership have same size, can just flatten both
        _eval_attack(
            attack_scores=scores[:, :, 0].view(-1),
            attack_membership=shadow_membership_mask.view(-1),
            output_dir=output_dir,
            suffix=f"{prefix}_global_{score_name}",
        )

    # LiRA
    for is_augmented in (False, True):
        if is_augmented and num_augmentations == 1:
            continue  # no augmentation available (e.g., Split-AI)
        logger.info(f"## LiRA {'w/' if is_augmented else 'w/o'} data augmentation")
        attack_suffix = "lira_da" if is_augmented else "lira"
        if is_augmented:
            shadow_attack_data = {
                score_name: attack_util.lira_attack_loo(
                    shadow_scores=scores,
                    shadow_membership_mask=shadow_membership_mask,
                )
                for score_name, scores in shadow_scores.items()
            }
        else:
            shadow_attack_data = {
                score_name: attack_util.lira_attack_loo(
                    shadow_scores=scores[:, :, 0].unsqueeze(-1),
                    shadow_membership_mask=shadow_membership_mask,
                )
                for score_name, scores in shadow_scores.items()
            }

        for score_name, (scores, membership) in shadow_attack_data.items():
            logger.info(f"### {score_name}")
            _eval_attack(
                attack_scores=scores,
                attack_membership=membership,
                output_dir=output_dir,
                suffix=f"{prefix}_{attack_suffix}_{score_name}",
            )


def _eval_attack(
    attack_scores: torch.Tensor,
    attack_membership: torch.Tensor,
    output_dir: pathlib.Path,
    suffix: str = "",
) -> None:
    score_file = output_dir / f"attack_scores_{suffix}.pt"
    torch.save(attack_scores, score_file)
    membership_file = output_dir / f"attack_membership_{suffix}.pt"
    torch.save(attack_membership, membership_file)

    # Calculate TPR at various FPR
    fpr, tpr, _ = sklearn.metrics.roc_curve(
        y_true=attack_membership.int().numpy(), y_score=attack_scores.numpy()
    )
    target_fprs = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
    for target_fpr in target_fprs:
        logger.info(
            f"TPR at FPR {target_fpr*100}%: {tpr[fpr <= target_fpr][-1]*100:.4f}%"
        )

    # Calculate attack accuracy
    prediction_threshold = torch.median(attack_scores).item()
    pred_membership = (
        attack_scores > prediction_threshold
    )  # median returns lower of two values => strict ineq.
    balanced_accuracy = torch.mean(
        (pred_membership == attack_membership).float()
    ).item()
    logger.info(f"Attack accuracy: {balanced_accuracy:.4f}")


def _train_model(
    train_data: data.Dataset,
    training_seed: int,
    verbose: bool = False,
) -> torch.nn.Module:
    # Same HPs as in original paper
    batch_size = 256
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4

    model = models.WideResNet(
        in_channels=3,
        depth=16,
        widen_factor=4,
        num_classes=10,
        use_group_norm=False,
        device=base.DEVICE,
        dtype=base.DTYPE,
    )

    train_datapipe = train_data.build_datapipe(
        shuffle=True,
        add_sharding_filter=True,
    )
    train_datapipe = train_datapipe.map(
        torchvision.transforms.v2.Compose(
            [
                torchvision.transforms.v2.RandomCrop(32, padding=4),
                torchvision.transforms.v2.RandomHorizontalFlip(),
                torchvision.transforms.v2.ToDtype(base.DTYPE, scale=True),
                torchvision.transforms.v2.Normalize(
                    mean=data.CIFAR10_MEAN,
                    std=data.CIFAR10_STD,
                ),
            ]
        ),
        input_col=0,
        output_col=0,
    )
    train_datapipe = train_datapipe.batch(batch_size, drop_last=False).collate()

    train_loader = torchdata.dataloader2.DataLoader2(
        train_datapipe,
        reading_service=torchdata.dataloader2.MultiProcessingReadingService(
            num_workers=int(os.environ.get("TORCH_DATALOADER_NUM_WORKER", "4")),
        ),
    )
    # This is a bit of a hack to ensure that the # steps per epoch is correct
    #  Using the length of the training datapipe might be wrong due to sharding and multiprocessing
    num_steps_per_epoch = 0
    for _ in train_loader:
        num_steps_per_epoch += 1
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    warnings.filterwarnings(
        action="ignore", category=UserWarning, module="torch.optim.lr_scheduler"
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / num_steps_per_epoch,
                end_factor=1.0,
                total_iters=num_steps_per_epoch,
            ),
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[epoch * num_steps_per_epoch for epoch in (60, 120, 160)],
                gamma=0.2,
            ),
        ],
        milestones=[1 * num_steps_per_epoch],
    )

    loss = torch.nn.CrossEntropyLoss()

    model.train()
    rng_loader_seeds = np.random.default_rng(seed=training_seed)
    for epoch in (pbar := tqdm.trange(NUM_TRAIN_EPOCHS, desc="Training", unit="epoch")):
        num_samples = 0
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        train_loader.seed(
            int(rng_loader_seeds.integers(0, 2**32, dtype=np.uint32, size=()))
        )
        for batch_xs, batch_targets in tqdm.tqdm(
            train_loader,
            desc="Current epoch",
            unit="batch",
            leave=False,
            disable=not verbose,
            total=num_steps_per_epoch,
        ):
            batch_xs = batch_xs.to(base.DEVICE)
            batch_targets = batch_targets.to(base.DEVICE)

            if batch_targets.dim() == 1:
                # Hard labels
                batch_ys = batch_targets
            else:
                batch_ys = torch.argmax(batch_targets, dim=-1)

            optimizer.zero_grad()
            batch_pred = model(batch_xs)
            batch_loss = loss(input=batch_pred, target=batch_targets)

            batch_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += batch_loss.item() * batch_xs.size(0)
            epoch_accuracy += (batch_pred.argmax(-1) == batch_ys).int().sum().item()
            num_samples += batch_xs.size(0)
        epoch_loss /= num_samples
        epoch_accuracy /= num_samples
        progress_dict = {
            "epoch_loss": epoch_loss,
            "epoch_accuracy": epoch_accuracy,
            "lr": lr_scheduler.get_last_lr()[0],
        }
        mlflow.log_metrics(progress_dict, step=epoch + 1)

        pbar.set_postfix(progress_dict)

    model.eval()
    return model


def _predict(
    model: torch.nn.Module,
    datapipe: torchdata.datapipes.iter.IterDataPipe,
    data_augmentation: bool,
    disable_tqdm: bool = False,
) -> torch.Tensor:
    # NB: Always returns data-augmentation dimension
    model.eval()
    datapipe = datapipe.map(torchvision.transforms.v2.ToDtype(base.DTYPE, scale=True))
    normalize_transform = torchvision.transforms.v2.Normalize(
        mean=data.CIFAR10_MEAN,
        std=data.CIFAR10_STD,
    )
    if not data_augmentation:
        # Augmentations add normalization later
        datapipe = datapipe.map(normalize_transform)
    datapipe = datapipe.batch(base.EVAL_BATCH_SIZE, drop_last=False).collate()
    pred_logits = []
    with torch.no_grad():
        for batch_xs in tqdm.tqdm(
            datapipe, desc="Predicting", unit="batch", disable=disable_tqdm
        ):
            if not data_augmentation:
                pred_logits.append(
                    model(batch_xs.to(dtype=base.DTYPE, device=base.DEVICE))
                    .cpu()
                    .unsqueeze(1)
                )
            else:
                flip_augmentations = (False, True)
                shift_augmentations = (0, -4, 4)
                batch_xs_pad = torchvision.transforms.v2.functional.pad(
                    batch_xs,
                    padding=[4],
                )
                pred_logits_current = []
                for flip in flip_augmentations:
                    for shift_y in shift_augmentations:
                        for shift_x in shift_augmentations:
                            offset_y = shift_y + 4
                            offset_x = shift_x + 4
                            batch_xs_aug = batch_xs_pad[
                                :, :, offset_y : offset_y + 32, offset_x : offset_x + 32
                            ]
                            if flip:
                                batch_xs_aug = (
                                    torchvision.transforms.v2.functional.hflip(
                                        batch_xs_aug
                                    )
                                )
                            # Normalization did not happen before; do it here
                            batch_xs_aug = normalize_transform(batch_xs_aug)
                            pred_logits_current.append(
                                model(
                                    batch_xs_aug.to(
                                        dtype=base.DTYPE, device=base.DEVICE
                                    )
                                ).cpu()
                            )
                pred_logits.append(torch.stack(pred_logits_current, dim=1))
    return torch.cat(pred_logits, dim=0)


class SplitAIHandler(object):
    def __init__(
        self,
        setting_seed: int,
        num_teachers: int,
        num_query: int,
        full_data: data.Dataset,
        membership_mask: torch.Tensor,
        canary_mask: torch.Tensor,
        poison_mask: torch.Tensor,
    ):
        # NB: Seed should be different for different shadow models, but the same for different teachers
        self._setting_seed = setting_seed

        assert 0 < num_query < num_teachers
        self._num_teachers = num_teachers
        self._num_query = num_query
        self._full_data = full_data
        self._membership_mask = membership_mask
        self._canary_mask = canary_mask
        self._poison_mask = poison_mask

        # Build full query sets for members and non-members
        rng = np.random.default_rng(seed=self._setting_seed)

        # IN: randomly select model indices
        #  Store them primarily for inference on test data later
        rng_query_in, rng = rng.spawn(2)
        # Populate iteratively to have the same query indices on original samples w/ and w/o duplicates
        num_in_samples = self._membership_mask.int().sum()
        self._query_indices_in = torch.empty(
            (num_in_samples, self._num_query), dtype=torch.long
        )
        for sample_idx in range(num_in_samples):
            self._query_indices_in[sample_idx] = torch.from_numpy(
                rng.permutation(self._num_teachers)[: self._num_query]
            )
        del rng_query_in

        # OUT: Pick a random query set from the training sample sets
        rng_query_out, rng = rng.spawn(2)
        self._query_indices_full = torch.empty(
            (len(self._full_data), self._num_query), dtype=torch.long
        )
        self._query_indices_full[self._membership_mask] = self._query_indices_in
        for sample_idx in range(len(self._full_data)):
            if not self._membership_mask[sample_idx]:
                self._query_indices_full[sample_idx] = self._query_indices_in[
                    rng_query_out.integers(0, num_in_samples)
                ]
        del rng_query_out

        # Store rng for inference on test data to be reproducible
        rng_test_inference, rng = rng.spawn(2)
        self._test_inference_seed = rng_test_inference.integers(
            0, 2**32, dtype=np.uint32
        )
        del rng_test_inference

        del rng

    def build_teacher_train_data(self, teacher_idx: int) -> data.Dataset:
        teacher_membership_mask, _ = self.get_teacher_masks(teacher_idx)
        teacher_membership_indices = torch.nonzero(teacher_membership_mask).squeeze(-1)
        return self._full_data.subset(teacher_membership_indices)

    def get_full_train_data(self) -> data.Dataset:
        return self._full_data

    def get_teacher_masks(
        self, teacher_idx: int
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        teacher_query_mask = torch.any(self._query_indices_full == teacher_idx, dim=1)

        # Select all samples that are members and do NOT use teacher for querying
        assert teacher_query_mask.size() == self._membership_mask.size()
        teacher_membership_mask = self._membership_mask & (~teacher_query_mask)
        assert len(self._full_data) == len(teacher_membership_mask)

        return teacher_membership_mask, self._canary_mask

    @property
    def num_teachers(self) -> int:
        return self._num_teachers

    def evaluate_ensemble(
        self, teacher_predictions: torch.Tensor, targets: torch.Tensor
    ) -> typing.Dict[str, float]:
        ensemble_probabilities = torch.mean(
            torch.softmax(teacher_predictions, dim=-1), dim=1
        )
        correct_predictions = torch.eq(
            torch.argmax(ensemble_probabilities, dim=-1), targets
        ).to(dtype=base.DTYPE_EVAL)
        return {
            "ensemble_test_accuracy": torch.mean(correct_predictions).item(),
        }

    def evaluate_splitai_test(
        self, teacher_predictions: torch.Tensor, targets: torch.Tensor
    ) -> typing.Dict[str, float]:
        rng = np.random.default_rng(seed=self._test_inference_seed)

        # None of the test samples are actually training members, hence sample query indices for all of them
        query_indices = torch.from_numpy(
            rng.choice(self._query_indices_in, size=len(targets), replace=True)
        )
        splitai_probabilities = self._splitai_inference(
            teacher_predictions, query_indices
        )
        correct_predictions = torch.eq(
            torch.argmax(splitai_probabilities, dim=-1), targets
        ).to(dtype=base.DTYPE_EVAL)
        return {
            "splitai_test_accuracy": torch.mean(correct_predictions).item(),
        }

    def infer_splitai_train_full(
        self,
        teacher_predictions: torch.Tensor,
    ) -> torch.Tensor:
        assert teacher_predictions.size(0) == len(self._full_data)
        return self._splitai_inference(
            teacher_predictions,
            self._query_indices_full,
            return_unreduced_logits=True,
        )

    def build_distillation_dataset(
        self, splitai_train_probs_full: torch.Tensor
    ) -> data.Dataset:
        return self._full_data.subset(
            indices=torch.nonzero(self._membership_mask).squeeze(-1),
            labels=splitai_train_probs_full[self._membership_mask],
        )

    def _splitai_inference(
        self,
        teacher_predictions: torch.Tensor,
        query_indices: torch.Tensor,
        return_unreduced_logits: bool = False,
    ):
        # Get raw log probs
        query_indices = query_indices.unsqueeze(-1).tile(
            (1, 1, teacher_predictions.size(-1))
        )
        assert teacher_predictions.size(0) == query_indices.size(0)
        raw_logits = torch.gather(teacher_predictions, dim=1, index=query_indices)
        assert raw_logits.size() == (
            teacher_predictions.size(0),
            query_indices.size(1),
            teacher_predictions.size(-1),
        )

        if return_unreduced_logits:
            return raw_logits
        else:
            return self.softlabels_from_raw(raw_logits)

    @classmethod
    def softlabels_from_raw(
        cls, splitai_unreduced_log_probs: torch.Tensor
    ) -> torch.Tensor:
        assert splitai_unreduced_log_probs.dim() >= 3
        return torch.mean(torch.softmax(splitai_unreduced_log_probs, dim=-1), dim=-2)

    def get_splitai_eval_data(
        self,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        membership_mask = torch.clone(self._membership_mask)
        canary_mask = torch.clone(self._canary_mask)
        poison_mask = torch.clone(self._poison_mask)
        train_ys = self._full_data.targets

        return membership_mask, train_ys, canary_mask, poison_mask


class DirectoryManager(object):
    def __init__(
        self,
        experiment_base_dir: pathlib.Path,
        experiment_name: str,
        run_suffix: typing.Optional[str] = None,
    ) -> None:
        self._experiment_base_dir = experiment_base_dir
        self._experiment_dir = self._experiment_base_dir / experiment_name
        self._run_suffix = run_suffix

    def get_training_teacher_output_dir(
        self, shadow_model_idx: int, teacher_idx: int
    ) -> pathlib.Path:
        return (
            self._get_shadow_model_base_dir(shadow_model_idx)
            / "teacher"
            / str(teacher_idx)
        )

    def get_training_student_output_dir(self, shadow_model_idx: int) -> pathlib.Path:
        return self._get_shadow_model_base_dir(shadow_model_idx)

    def _get_shadow_model_base_dir(self, shadow_model_idx: int) -> pathlib.Path:
        actual_suffix = "" if self._run_suffix is None else f"_{self._run_suffix}"
        return self._experiment_dir / ("shadow" + actual_suffix) / str(shadow_model_idx)

    def get_training_log_dir(
        self, shadow_model_idx: int, teacher_idx: typing.Optional[int]
    ) -> pathlib.Path:
        # Log all MLFlow stuff into the same directory, for all experiments!
        return self._experiment_base_dir / "mlruns"

    def get_attack_output_dir(self) -> pathlib.Path:
        return (
            self._experiment_dir
            if self._run_suffix is None
            else self._experiment_dir / f"attack_{self._run_suffix}"
        )


def parse_args() -> argparse.Namespace:
    default_data_dir = pathlib.Path(os.environ.get("DATA_ROOT", "data"))
    default_base_experiment_dir = pathlib.Path(os.environ.get("EXPERIMENT_DIR", ""))

    parser = argparse.ArgumentParser()

    # General args
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=default_data_dir,
        help="Dataset root directory",
    )
    parser.add_argument(
        "--experiment-dir",
        default=default_base_experiment_dir,
        type=pathlib.Path,
        help="Experiment directory",
    )
    parser.add_argument("--experiment", type=str, default="dev", help="Experiment name")
    parser.add_argument(
        "--run-suffix",
        type=str,
        default=None,
        help="Optional run suffix to distinguish multiple runs in the same experiment",
    )
    parser.add_argument("--verbose", action="store_true")

    # Dataset and setup args
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--num-shadow", type=int, default=64, help="Number of shadow models"
    )
    parser.add_argument(
        "--num-canaries", type=int, default=500, help="Number of canaries to audit"
    )
    parser.add_argument(
        "--canary-type",
        type=data.CanaryType,
        default=data.CanaryType.CLEAN,
        choices=list(data.CanaryType),
        help="Type of canary to use",
    )
    parser.add_argument(
        "--num-poison", type=int, default=0, help="Number of poison samples to include"
    )
    parser.add_argument(
        "--poison-type",
        type=data.PoisonType,
        default=data.PoisonType.CANARY_DUPLICATES,
        choices=list(data.PoisonType),
        help="Type of poisoning to use",
    )

    # Defense-specific
    parser.add_argument(
        "--num-teachers", type=int, required=True, help="Number of ensemble members (K)"
    )
    parser.add_argument(
        "--num-query", type=int, required=True, help="Number of queries per sample (L)"
    )

    # Create subparsers per action
    subparsers = parser.add_subparsers(
        dest="action", required=True, help="Action to perform"
    )

    # Train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--shadow-model-idx",
        type=int,
        required=True,
        help="Train shadow model with index",
    )

    # Defense-specific
    train_subparser = train_parser.add_subparsers(
        dest="train_action", required=True, help="Mechanism to train"
    )
    # Train: Split-AI
    train_splitai_parser = train_subparser.add_parser("splitai")
    train_splitai_parser.add_argument(
        "--teacher-idx", type=int, required=True, help="Index of teacher to train"
    )
    # Train: Distillation
    train_distillation_parser = train_subparser.add_parser("distillation")  # noqa: F841

    # MIA
    attack_parser = subparsers.add_parser("attack")
    attack_parser.add_argument(
        "--embeddings-file",
        type=pathlib.Path,
        required=False,
        help="Embeddings to determine nearest neighbors",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
