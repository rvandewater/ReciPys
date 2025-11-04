---
title: >-
  `ReciPies`: A Lightweight Data Transformation Pipeline for Reproducible ML
authors:
  - name: Robin P. van de Water
    email: robin.vandewater@hpi.de
    affiliation: "1, 2"
    orcid: 0000-0002-2895-4872
    corresponding: true
  - name: Hendrik Schmidt
    orcid: 0000-0001-7699-3983
    affiliation: "1"
    equal-contrib: false
  - name: Patrick Rockenschaub
    orcid: 0000-0002-6499-7933
    affiliation: "3"
    equal-contrib: false
affiliations:
  - index: 1
    name: Hasso Plattner Institute, University of Potsdam, Potsdam, Germany
  - index: 2
    name: Hasso Plattner Institute for Digital Health at Mount Sinai, Icahn School of Medicine at Mount Sinai, New York City, NY, USA
  - index: 3
    name: Innsbruck Medical University, Innsbruck, Austria
date: 2025-07-25
bibliography: paper.bib
repository: https://github.com/rvandewater/ReciPies
tags:
---

# Summary

Machine Learning (ML) workflows live or die by their data‑preprocessing steps, yet in Python, these steps are often
scattered across ad‑hoc scripts or opaque Scikit-Learn (sklearn) snippets that are hard to read, audit, or reuse.
`ReciPies` provides a concise, human‑readable, and fully reproducible method to declare, execute, and share
preprocessing pipelines, that adheres to Configuration as Code principles.
It lets users describe transformations as a recipe made of ordered *steps* (e.g., imputing, encoding, normalizing)
applied to variables identified by semantic roles (predictor, outcome, ID, time stamp, etc.). Recipes can be *prepped*
once, *baked* many times, and separated between training and new data. `ReciPies` provides the choice of Pandas and
Polars backends and is easily extensible. Data provenance can be tracked and published.
Packaging preprocessing as clear, declarative objects, `ReciPies` lowers the cognitive load of feature engineering,
improves reproducibility, and makes methodological choices explicit, benefiting individual researchers, engineering
teams, and peer reviewers alike.

# Statement of need

Robust machine‑learning results in science hinge on transparent, reproducible data‑preprocessing—yet in Python, these
steps are typically spread across ad‑hoc notebooks or are buried inside opaque scripts; that is, if this code is even
made available. Additionally, most variable semantics are unclear (called *roles*). These problems confound research
results, complicate peer review and hinder reuse. Researchers and engineers working with longitudinal regulated data (
e.g., energy production, finance, and environmental monitoring) especially need pipelines they are able to audit,
serialize, and hand to collaborators without reverse‑engineering a tangle of imperative code [@10.1145/3641525]. The
lack of reproducibility has been documented extensively in
literature [@johnsonReproducibilityCriticalCare2017a; @kellyKeyChallengesDelivering2019a; @semmelrockReproducibilityMachinelearningbasedResearch2025].

`ReciPies` fills this gap by bringing a tidy, stepwise *recipe* interface to Python. Users declare transformations over
variables selected by semantic roles; recipes are "prepped" once on training data and "baked" on new data to eliminate
leakage; and every step is inspectable, versionable, and serializable (JSON/YAML). `Recipes` runs on
Pandas [@mckinney-proc-scipy-2010] and Polars [@vinkPolarsPolarsPython2024] for interoperability and performance, and their
Object-oriented abstractions enable users to implement custom steps. The framework is declarative and reproducible for
data preprocessing, prioritizing human readability and methodological transparency.
We demonstrate that there is no need to sacrifice readability for performance or flexibility for simplicity. By reducing
the cognitive overhead of feature engineering and making methodological choices explicit, `ReciPies` enables researchers
to focus on their core work. We hope this will broaden the reproducibility discussion in ML from hyperparameters to the
entire experiment pipeline. We encourage the development of domain-specific step libraries and integration patterns that
can benefit the broader ecosystem.

# Related Work

Our work brings the Recipes [@kuhnRecipesPreprocessingFeature2024] framework to Python and extends it for the ML
community. The design enables straightforward integration as part of a pipeline that includes ML libraries like
sklearn [@pedregosa_scikit-learn_2011] and PyTorch[@paszkePyTorchImperativeStyle2019]. To the best of our knowledge, no
other packages comply with the flexibility and reproducibility of `ReciPies` and its Configuration as Code approach.
Sklearn offers composable transformers, but no role-based variable grammar, limited human-readability, and awkward
serialization.
Feature-engine [@galliFeatureenginePythonPackage2021], pyjanitor [@j.PyjanitorCleanerAPI2019], or
scikit-lego[@warmerdamKoaningScikitlegoV0952025] add helpful transformers or cleaning verbs. However, none provide a
unified, declarative recipe abstraction with a strict "prep/bake" split and backend flexibility. `ReciPies` provides a
stepwise recipe that is easy to use and read, allowing users to readily preprocess data for a wide range of machine
learning pipelines.

# Usage

If we have a dataset, `df`, with a label `y`, some features `x1`, `x2`, `x3`, `x4`, an identifier `id`, and a sequential
component `time`, we can build a preprocessing pipeline using `ReciPies`. We first do a train/test split:

``` Python
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
```

We then define the roles of the variables in this dataset:

``` Python
roles = {outcomes:["y"], predictors=["x1", "x2", "x3", "x4"], groups=["id"], 
sequences=["time"]}
```

Afterward, we create the `ingredients` which encapsulate the training data and its roles, and the `recipe` to
preprocess the data:

``` Python
ing = Ingredients(df_train, roles=roles)
rec = Recipe(ing)
```

We add preprocessing steps:

``` Python
rec.add_step(StepScale())
rec.add_step(StepSklearn(MissingIndicator(features="all"), 
  sel=has_role("predictor")))
rec.add_step(StepImputeFill(strategy="forward"))
rec.add_step(StepSklearn(LabelEncoder(), sel=has_type("categorical"), 
  columnwise=True))
```

We can now fit the recipe and transform both the train and test set without leakage and in a transparent manner:

``` Python
df_train = rec.prep()
df_test = rec.bake(df_test)
```

We can use the `bake` method on the training set to transform it again, e.g., to apply the same transformations to
a new dataset. Complete code, benchmarks, and interactive notebooks are available in the project documentation.
`ReciPies` also provides a benchmarking suite with results to compare the performance of different preprocessing steps
on (generated) data.

`ReciPies` is used as the bedrock of reproducible pipelines of Yet Another ICU
Benchmark [@vandewaterAnotherICUBenchmark2024a]
The adaptable, configurable code modules that make extensive use of `ReciPies` can be
found [here](https://github.com/rvandewater/YAIB/blob/development/icu_benchmarks/data/preprocessor.py); this
demonstrates that `ReciPies` can be used for arbitrary research domains.

# Future steps

We plan to expand the library of Polars-native steps to fully leverage its columnar execution model, particularly for
time-series operations and large-scale aggregations, where Polars shows significant performance advantages. We envision
`ReciPies` recipes as portable preprocessing artifacts that can be versioned, tracked, and deployed across different
environments. Tighter integration with experiment tracking and model registries would streamline the transition from
research to production, a complex process in many application domains.
