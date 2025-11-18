---
title: >-
  `ReciPies`: A Lightweight Data Transformation Pipeline for Reproducible ML
authors:
  - name: Robin P. van de Water
    email: robin.vandewater@hpi.de
    affiliation: 1, 2
    orcid: 0000-0002-2895-4872
    corresponding: true
  - name: Hendrik Schmidt
    orcid: 0000-0001-7699-3983
    affiliation: '1'
    equal-contrib: false
  - name: Patrick Rockenschaub
    orcid: 0000-0002-6499-7933
    affiliation: '3'
    equal-contrib: false
affiliations:
  - index: 1
    name: Hasso Plattner Institute, University of Potsdam, Potsdam, Germany
  - index: 2
    name: Hasso Plattner Institute for Digital Health at Mount Sinai, Icahn School of Medicine at Mount Sinai, New York City, NY, USA
  - index: 3
    name: Innsbruck Medical University, Innsbruck, Austria
date: 2025-11-13
bibliography: paper.bib
repository: https://github.com/rvandewater/ReciPies
tags:
---

# Summary

Machine Learning (ML) workflows live or die by their data‑preprocessing steps. In Python, these steps are often
scattered across ad‑hoc scripts or opaque scikit-learn snippets that are hard to read, audit, or reuse.
`ReciPies` provides a concise, human‑readable, and reproducible way to declare, execute, and share
preprocessing pipelines following configuration-as-code principles. It lowers the cognitive load of
feature engineering, improves reproducibility, and makes methodological choices explicit for
researchers, engineering teams, and peer reviewers.

# Statement of need

Transparent and reproducible preprocessing remains a weak link in many scientific ML studies. The consequences are (1) confounded research results,
(2) complicated peer review, and (3) poor reuse. Researchers and engineers working with longitudinal regulated data (e.g., in energy production, health, finance, or environmental monitoring) in particular need pipelines they can audit,
serialize, and hand to collaborators without reverse‑engineering a tangle of imperative code [@10.1145/3641525]. The
current lack of reproducibility has been documented extensively in the
literature [@johnsonReproducibilityCriticalCare2017a; @kellyKeyChallengesDelivering2019a; @semmelrockReproducibilityMachinelearningbasedResearch2025].

# Related work

Scikit-learn provides `Pipeline` and `ColumnTransformer`, along with a rich estimator ecosystem [@pedregosa_scikit-learn_2011], but lacks role-based variable grammar, limited human readability, and awkward serialization. Feature-engine [@galliFeatureenginePythonPackage2021], pyjanitor [@j.PyjanitorCleanerAPI2019], or
scikit-lego[@warmerdamKoaningScikitlegoV0952025] add helpful transformers and data-cleaning verbs. However, none provide a
unified, role-centric abstraction with backend flexibility. The R `recipes` package established the prep/bake pattern and a clean grammar for preprocessing [@kuhnRecipesPreprocessingFeature2024]. `ReciPies` brings these ideas to Python, extends them with backend-agnostic execution on Pandas and Polars, and emphasizes configuration-as-code artifacts suitable for a wide range of machine
learning pipelines.

# Design and implementation

`ReciPies` adopts a tidy, stepwise *recipe* interface that emphasizes semantic roles over column names and a strict separation of fitting from application. Transformations are declared on roles such as predictor, outcome, identifier, or timestamp. Recipes are prepped on training data and baked on new data to prevent leakage. Each step is inspectable, versionable, and serializable to JSON or YAML for provenance and review. Steps are composable with explicit state and deterministic behavior given fixed inputs and seeds. `ReciPies` supports both Pandas [@mckinney-proc-scipy-2010], which is widely adopted in the ML community, and the more recent Polars [@vinkPolarsPolarsPython2024], which offers increased performance.
![recipies flowchart](../docs/figures/recipies_flow.pdf)
A typical workflow 1) loads a Pandas or Polars training `DataFrame`, 2) wraps it as an `Ingredients` object that records role metadata, 3) defines a `Recipe` from `Steps` operating on columns selected based on roles by `Selectors`, 4) preps the recipe on the training split to estimate parameters, and 5) bakes it on the held-out split to apply those parameters without leakage. The baked outputs feed downstream modeling and evaluation. Figure 1 gives an overview of this workflow.
![physionet code snippet](../docs/figures/recipies_code_snippet.pdf)
Figure 2 demonstrates usage on the PhysioNet Computing in Cardiology 2019 dataset [@reynaEarlyPredictionSepsis2020a], including role assignment, temporal imputation, and normalization. The prepped recipe serializes to JSON or YAML, and reloading the artifact reproduces the transforms across supported platforms.

Complete code and interactive notebooks are available in the project documentation. `ReciPies` also provides a benchmarking suite comparing the performance of different preprocessing steps
on (generated) data.
`ReciPies` is already used as the bedrock of reproducible pipelines in Yet Another ICU
Benchmark [@vandewaterAnotherICUBenchmark2024a]
The adaptable, configurable code modules that make extensive use of `ReciPies` can be
found [here](https://github.com/rvandewater/YAIB/blob/development/icu_benchmarks/data/preprocessor.py); this
demonstrates that `ReciPies` can be used for arbitrary research domains. Our work shows that there is no need to sacrifice readability for performance, nor flexibility for simplicity. We encourage the development of domain-specific step libraries and integration patterns that
can benefit the broader ecosystem.

# Future steps

Our first step is to expand the library of Polars-native steps to fully leverage its columnar execution model, particularly for
time-series operations and large-scale aggregations, where Polars shows significant performance advantages. Second, we aim to integrate with ML versioning systems to streamline the
transition from research to production.

# Acknowledgements
Robin P. van de Water is funded by the European Commission in the Horizon 2020 project INTERVENE (Grant agreement ID: 101016775).
This work has been edited with the help of Large Language Models (LLMs) to improve readability. 

# References