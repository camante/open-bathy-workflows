import pandas as pd
import pytest

import train


def test_stratified_train_test_split_fails_closed_on_unstratifiable_depths():
    df = pd.DataFrame({"depth_m": [1.0]})
    with pytest.raises(ValueError, match="Stratified train/test split failed"):
        train.stratified_train_test_split(df, target_col="depth_m", test_size=0.5, seed=1)
