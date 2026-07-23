import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # MIRA claim-by-claim reproduction

        **Evidence first:** five claims are `VERIFIED`; the 13-dimensional
        physical-lensing claim is `BLOCKED` because its raw posterior inputs
        and essential protocol details were not released. This notebook embeds
        the small final results, so opening it does not rerun expensive work.
        """
    )
    return


@app.cell
def _(mo):
    claim_rows = [
        {"claim": 1, "verdict": "VERIFIED", "evidence": "exact null mass + implementation parity"},
        {"claim": 2, "verdict": "VERIFIED", "evidence": "Laplace equation cells"},
        {"claim": 3, "verdict": "VERIFIED", "evidence": "N=5,000 asymptotic reference"},
        {"claim": 4, "verdict": "VERIFIED", "evidence": "exact identity + 6,435 monotone maps"},
        {"claim": 5, "verdict": "BLOCKED", "evidence": "six release-completeness blockers"},
        {"claim": 6, "verdict": "VERIFIED", "evidence": "exact L=16, d=12,288 tensors"},
    ]
    mo.ui.table(claim_rows, selection=None, label="Terminal claim evidence")
    return (claim_rows,)


@app.cell
def _(mo):
    paper_scores = [0.6442, 0.5783, 0.5298, 0.5056]
    observed_scores = [0.6830859375, 0.5097265625, 0.503212890625, 0.4953125]
    labels = ["true", "noise shift", "prior shift", "both shifts"]
    rows = [
        {
            "condition": label,
            "paper": paper,
            "observed": observed,
            "observed_minus_paper": observed - paper,
        }
        for label, paper, observed in zip(labels, paper_scores, observed_scores)
    ]
    mo.vstack(
        [
            mo.md(
                """
                ## Claim 6: exact released galaxy tensors

                The complete order is reproduced at `L=16`, `N=64`, and
                12,288 dimensions. The largest numerical disagreement is kept
                visible rather than relabeled as exact agreement.
                """
            ),
            mo.ui.table(rows, selection=None),
        ]
    )
    return labels, observed_scores, paper_scores, rows


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Why MIRA tends toward 2/3

        If candidate and truth match, the random region mass is uniform. After
        Laplace smoothing, the score distribution converges to `Beta(2, 1)`,
        whose mean is `2/3` and variance is `1/18`.

        The finite lower-bound identity used in Claim 4 is

        \[
        \mu_{\mathrm{MIRA}}
        = \frac{1}{2} + \frac{N}{N+2}
          \left(2\,\mathbb{E}[U T(U)]-\mathbb{E}[T(U)]\right).
        \]

        Under the proposition's continuous radial-CDF assumptions, `T` is
        nondecreasing, so the bracket is nonnegative.
        """
    )
    return


@app.cell
def _(mo):
    intact = 0.6830859375
    rolled = 0.518330078125
    controls = [
        {"control": "full-scale released-code parity", "result": "absolute delta 0.0", "passed": True},
        {"control": "truth pairing rolled by one", "result": f"drop {intact - rolled:.4f}", "passed": True},
        {"control": "norm=True ambiguity sibling", "result": "true first; remaining order differs", "passed": True},
        {"control": "Claim 5 fake zero-byte assets", "result": "rejected", "passed": True},
    ]
    mo.vstack(
        [
            mo.md("## Sensitivity checks"),
            mo.ui.table(controls, selection=None),
        ]
    )
    return controls, intact, rolled


@app.cell
def _(mo):
    missing = [
        "L=100, d=13 truth parameters",
        "four 100×20,000×13 posterior sets",
        "MALA source, step size, initialization, tuning",
        "executable caustics EPL/SIE–Sérsic construction",
        "values for parameters described as held constant",
        "dependency versions and random seeds",
    ]
    mo.callout(
        mo.md(
            "## Claim 5 is BLOCKED\n\n"
            + "\n".join(f"- {item}" for item in missing)
            + "\n\n`BLOCKED` does not mean the paper claim is false. It means a "
            "faithful verification or falsification cannot be constructed from "
            "the public record."
        ),
        kind="warn",
    )
    return (missing,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Reproduce locally

        ```bash
        uv sync --frozen
        uv run --frozen python repro/src/run_campaign.py
        ```

        Formal experiments use that exact command on every branch. The report
        beside this notebook contains source hashes, branch links, runtime,
        deviations, and the publication gate. No Hugging Face revision was
        published without explicit approval.
        """
    )
    return


if __name__ == "__main__":
    app.run()
