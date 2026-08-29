"""Command line entry point: ``faspy <command>``.

Run ``faspy --help`` for the list, or ``faspy <command> --help`` for one
command's options. Every default comes from :mod:`faspy.config`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    CELLPOSE_EPOCHS,
    CELLPOSE_RESCALE,
    CELLPROB_THRESHOLD,
    EPOCHS,
    FLOW_THRESHOLD,
    N_FOLDS,
)


def _prepare(args):
    from . import datasets

    if not args.skip_dic:
        datasets.convert_dic()
    if not args.skip_examples:
        datasets.integrate_examples()
    datasets.build_dataset()


def _evaluate(args):
    from . import evaluate

    if args.route == "zeroshot":
        evaluate.zero_shot_baseline(
            rescale=args.rescale, cellprob=args.cellprob, flow=args.flow, limit=args.limit
        )
    elif args.route == "instances":
        evaluate.cross_validate_instances(
            n_folds=args.folds,
            epochs=args.epochs,
            rescale=args.rescale,
            cellprob=args.cellprob,
            flow=args.flow,
            train_final=not args.no_final,
        )
    else:
        evaluate.cross_validate_semantic(
            n_folds=args.folds, epochs=args.epochs, train_final=not args.no_final
        )


def _quantify(args):
    from . import quantify

    quantify.run(model_name=args.model, rescale=args.rescale, limit=args.limit)


def _figure(args):
    from . import figures

    if args.subject == "diameters":
        figures.diameter_figure(output=args.output, limit=args.limit)
        return

    if not args.key:
        raise SystemExit(f"figure {args.subject}: a section key is required")
    render = figures.traits_figure if args.subject == "traits" else figures.pipeline_figure
    render(key=args.key, output=args.output, model_name=args.model, rescale=args.rescale)


def _diagnose(args):
    from . import diagnostics

    if args.check == "images":
        # Le compte de fichiers illisibles doit remonter jusqu'au code de sortie.
        # Un diagnostic qui signale un probleme et rend 0 n'est vu par aucune
        # verification automatisee.
        if diagnostics.check_images():
            raise SystemExit(1)
    elif args.check == "annotations":
        diagnostics.check_annotations()
    elif args.check == "orphans":
        diagnostics.orphan_census(
            model_name=args.model,
            folds=args.folds,
            rescale=args.rescale,
            limit=args.limit,
            crops=not args.no_crops,
        )
    elif args.check == "sweep":
        diagnostics.sweep_thresholds(
            fold=args.fold,
            folds=args.folds,
            model_name=args.model,
            cellprob_values=args.cellprob,
            flow_values=args.flow,
            rescale=args.rescale,
            limit=args.limit,
        )
    elif args.check == "depth":
        diagnostics.calibrate_depth()
    elif args.check == "lumen":
        diagnostics.compare_lumen_masks()
        if args.reference:
            print()
            diagnostics.validate_lumen(args.reference)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="faspy", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare", help="convert the sources and build the derived dataset"
    )
    prepare.add_argument("--skip-dic", action="store_true", help="do not convert the DIC set")
    prepare.add_argument(
        "--skip-examples", action="store_true", help="do not copy the reference sections"
    )
    prepare.set_defaults(handler=_prepare)

    evaluate = commands.add_parser("evaluate", help="run cross-validation")
    evaluate.add_argument(
        "route", choices=("instances", "semantic", "zeroshot"),
        help="instances: fine-tuned Cellpose-SAM; semantic: U-Net; "
             "zeroshot: the published checkpoint with no fine-tuning, the "
             "baseline the fine-tuned model has to beat",
    )
    evaluate.add_argument("--folds", type=int, default=N_FOLDS)
    evaluate.add_argument(
        "--epochs", type=int, default=None, help="default: 100 for instances, 35 for semantic"
    )
    evaluate.add_argument("--rescale", type=float, default=CELLPOSE_RESCALE)
    evaluate.add_argument("--cellprob", type=float, default=CELLPROB_THRESHOLD)
    evaluate.add_argument("--flow", type=float, default=FLOW_THRESHOLD)
    evaluate.add_argument(
        "--no-final", action="store_true", help="skip the final model trained on everything"
    )
    evaluate.add_argument("--limit", type=int, default=0, help="zeroshot only: cap the sections")
    evaluate.set_defaults(handler=_evaluate)

    quantify = commands.add_parser("quantify", help="quantify every section for production")
    quantify.add_argument("--model", default="cpsam_final")
    quantify.add_argument("--rescale", type=float, default=CELLPOSE_RESCALE)
    quantify.add_argument("--limit", type=int, default=0, help="0 = every section")
    quantify.set_defaults(handler=_quantify)

    figure = commands.add_parser("figure", help="render a figure")
    figure.add_argument(
        "subject", choices=("pipeline", "traits", "diameters"),
        help="pipeline: the walk-through for one section; "
             "traits: what every anatomical trait measures, drawn on the section; "
             "diameters: bundle size against the range Cellpose-SAM was pretrained on",
    )
    figure.add_argument("key", nargs="?", default=None, help="section key, e.g. GALB_0061_1")
    figure.add_argument("--limit", type=int, default=0, help="diameters only: cap the sections")
    figure.add_argument("--model", default=None,
                        help="checkpoint to segment with; omitted, the panel shows the annotation")
    figure.add_argument("--rescale", type=float, default=CELLPOSE_RESCALE)
    figure.add_argument("--output", type=Path, default=None)
    figure.set_defaults(handler=_figure)

    diagnose = commands.add_parser("diagnose", help="data quality and audit reports")
    diagnose.add_argument(
        "check",
        choices=("images", "annotations", "orphans", "sweep", "depth", "lumen"),
        help="images: unreadable files; annotations: implausible masks; "
             "orphans: detected but unannotated bundles; sweep: decoding "
             "thresholds; depth: trichome filter calibration; lumen: compare "
             "areas with manual measurements and calibrate the brightness threshold",
    )
    diagnose.add_argument(
        "--model", default=None, help="single checkpoint; omit for out-of-fold prediction"
    )
    diagnose.add_argument("--fold", type=int, default=0)
    diagnose.add_argument("--folds", type=int, default=N_FOLDS)
    diagnose.add_argument("--rescale", type=float, default=CELLPOSE_RESCALE)
    diagnose.add_argument("--cellprob", type=float, nargs="+", default=[0.0, -1.0, -2.0, -3.0])
    diagnose.add_argument("--flow", type=float, nargs="+", default=[FLOW_THRESHOLD])
    diagnose.add_argument("--limit", type=int, default=0)
    diagnose.add_argument("--no-crops", action="store_true")
    diagnose.add_argument("--reference", type=Path, default=None,
                          help="lumen only: CSV of manual area measurements")
    diagnose.set_defaults(handler=_diagnose)

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate" and args.epochs is None:
        args.epochs = CELLPOSE_EPOCHS if args.route == "instances" else EPOCHS
    args.handler(args)


if __name__ == "__main__":
    main()
