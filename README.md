# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/miranska/quantum-mutant-planner/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                              |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/quantum\_mutant\_planner/\_\_init\_\_.py      |        5 |        0 |        0 |        0 |    100% |           |
| src/quantum\_mutant\_planner/analysis.py          |      187 |        0 |       82 |        1 |     99% | 296-\>300 |
| src/quantum\_mutant\_planner/certificates.py      |      198 |        2 |       78 |        2 |     99% |  418, 425 |
| src/quantum\_mutant\_planner/cli.py               |       34 |        0 |        2 |        1 |     97% |  98-\>103 |
| src/quantum\_mutant\_planner/conf/\_\_init\_\_.py |        0 |        0 |        0 |        0 |    100% |           |
| src/quantum\_mutant\_planner/examples.py          |       20 |        0 |        2 |        0 |    100% |           |
| src/quantum\_mutant\_planner/models.py            |      123 |        0 |        0 |        0 |    100% |           |
| src/quantum\_mutant\_planner/mutations.py         |       94 |        1 |       38 |        1 |     98% |        39 |
| src/quantum\_mutant\_planner/pauli.py             |       56 |        0 |       30 |        1 |     99% |   81-\>98 |
| src/quantum\_mutant\_planner/pilot.py             |       85 |        0 |       40 |        0 |    100% |           |
| src/quantum\_mutant\_planner/planner.py           |      219 |        2 |       94 |        2 |     99% |  231, 239 |
| src/quantum\_mutant\_planner/statistics.py        |       51 |        0 |       22 |        0 |    100% |           |
| **TOTAL**                                         | **1072** |    **5** |  **388** |    **8** | **99%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/miranska/quantum-mutant-planner/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/miranska/quantum-mutant-planner/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/miranska/quantum-mutant-planner/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/miranska/quantum-mutant-planner/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fmiranska%2Fquantum-mutant-planner%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/miranska/quantum-mutant-planner/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.