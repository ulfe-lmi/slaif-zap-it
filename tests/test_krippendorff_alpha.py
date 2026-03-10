import numpy as np
import pytest

from src.postprocessing import krippendorff_alpha_ordinal, krippendorf_alfa


def test_krippendorff_alpha_ordinal_perfect_agreement():
    ratings = np.array(
        [
            [0, 0, 0],
            [1, 1, np.nan],
            [2, 2, 2],
        ],
        dtype=float,
    )

    assert krippendorff_alpha_ordinal(ratings, categories=3) == pytest.approx(1.0)


def test_krippendorff_alpha_ordinal_three_category_disagreement():
    ratings = np.array(
        [
            [0, 2],
            [0, 2],
            [2, 0],
            [2, 0],
        ],
        dtype=float,
    )

    assert krippendorff_alpha_ordinal(ratings, categories=3) == pytest.approx(-0.75)


def test_krippendorf_alfa_alias_uses_three_ordinal_categories():
    ratings = np.array(
        [
            [0, 0],
            [1, 1],
            [2, 2],
        ],
        dtype=float,
    )

    assert krippendorf_alfa(ratings) == pytest.approx(1.0)
