from typing import Set
import irisclassifier
import pytest

performance_threshoulds = {0.50, 0.75, 0.95}

@pytest.mark.parametrize('th', performance_threshoulds)
def test_evaluation(th):
    i = irisclassifier.IrisClassifier()
    i.ingestion()
    i.segregation()
    i.train()
    res = i.evaluation()
    assert res > th