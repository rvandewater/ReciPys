---
title: >-
    `ReciPies`: A Lightweight Data Transformation Pipeline for Reproducible ML
authors:
  - name: Robin P. van de Water
    email: robin.vandewater@hpi.de
    affiliation: [1, 2]
    orcid: 0000-0002-2895-4872ß
    corresponding: true
  - name: Hendrik Schmidt
    orcid: 0000-0001-7699-3983
    affiliation: 
    equal-contrib: false
  - name: Patrick Rockenschaub
    orcid: 0000-0002-6499-7933
    affiliation: [1, 3]
    equal-contrib: false
affiliations:
  - index: 1
    name: Hasso Plattner Institute, University of Potsdam, Potsdam, Germany
  - index: 2
    name: Hasso Plattner Institute for Digital Health at Mount Sinai, Icahn School of Medicine at Mount Sinai, New York City, NY, USA
  - index: 3
    name: Innsbruck Medical University, Innsbruck, Austria
date: 2025-07-24
bibliography: paper.bib
repository: https://github.com/rvandewater/`ReciPies`
tags:
  - reference
  - example
  - markdown
  - publishing
---
<!-- 
Guide:
https://github.com/openjournals/inara/blob/main/example/paper.md  -->
<!-- # Summary

Machine learning pipelines often require complicated preprocessing pipelines. Our aim is to simplify this with a python package that can be used to define human-readable and reproducible pipelines for machine learning tasks.

# Statement of Need

Python is the most popular programming language for machine learning. However, preprocessing data for machine learning tasks can be a time-consuming and error-prone process. `ReciPies` is a python package that aims to simplify this process by providing a fast and intuitive interface for preprocessing data. It can use both a polars [@PolarsPolars2024] backend, which allows for fast python-native data processing, and a traditional Pandas [@mckinney-proc-scipy-2010] backend for use with legacy data tools. 
`ReciPies` is designed to be easy to use and flexible, allowing users to easily preprocess data for a wide range of machine learning pipelines. 
Moreover, we hide the complexity of the preprocessing pipeline from the user which allows for more reproducible and maintainable code. -->

# Summary
Modern machine learning (ML) workflows live or die by their data‑preprocessing steps, yet in Python—a language with a rich ecosystem for data science and ML—these steps are often scattered across ad‑hoc scripts or opaque Scikit-Learn (sklearn) snippets that are hard to read, audit, or reuse. `ReciPies` provides a concise, human‑readable, and fully reproducible way to declare, execute, and share preprocessing pipelines, adhering to Configuration as Code principles. It lets users describe transformations as a recipe made of ordered *steps* (e.g., imputing, encoding, normalizing) applied to variables identified by semantic roles (predictor, outcome, ID, time stamp, etc.). Recipes can be *prepped* (trained) once, *baked* many times, and cleanly separated between training and new data—preventing data leakage by construction. Under the hood, `ReciPies` targets both Pandas and Polars backends for performance and flexibility, and it is easily extensible: users can register custom steps with minimal boilerplate. Each recipe is serializable to JSON/YAML for provenance tracking, collaboration, and publication, and integrates smoothly with downstream modeling libraries. Packaging preprocessing as clear, declarative objects, `ReciPies` lowers the cognitive load of feature engineering, improves reproducibility, and makes methodological choices explicit, benefiting individual researchers, engineering teams, and peer reviewers alike.

# Statement of need
Robust machine‑learning results in science hinge on transparent, reproducible data‑preprocessing—yet in Python, these steps are typically spread across ad‑hoc notebooks or are buried inside opaque scripts; that is, if this code is even made available. Additionally, most variable semantics are unclear (*Which columns are outcomes? IDs? Time stamps?*); this encourages accidental data leakage when "fitting" transforms on the full dataset, confounding research results, complicating peer review, and hindering reuse. Researchers and engineers working with longitudinal or regulated data (e.g., energy production, finance, and environmental monitoring) especially need pipelines they are able to audit, serialize, and hand to collaborators without reverse‑engineering a tangle of imperative code [@10.1145/3641525]. The lack of reproducibility has been documented extensively in literature [@johnsonReproducibilityCriticalCare2017a; @kellyKeyChallengesDelivering2019a; @semmelrockReproducibilityMachinelearningbasedResearch2025].

`ReciPies` fills this gap by bringing a tidy, stepwise *recipe* interface to Python. Users declare transformations over variables selected by semantic roles; recipes are "prepped" once on training data and "baked" on new data to eliminate leakage; and every step is inspectable, versionable, and serializable (JSON/YAML). `Recipes` runs on Pandas [@mckinney-proc-scipy-2010] and Polars [@PolarsPolars2024] for performance and interoperability, and its object-oriented abstractions enable users to implement custom steps with minimal boilerplate. 

<!-- We illustrate its utility on an intensive‑care prediction task and in teaching settings, showing that a clear, declarative preprocessing grammar reduces cognitive load, eases collaboration, and strengthens the reproducibility of published ML results. -->

# Related Work
The R package by [@kuhnRecipesPreprocessingFeature2024] inspired our work and the concepts used in this package. As such, it is designed to be a Python equivalent of the R package, with improvements in functionality aimed at the ML community. The design enables 
straightforward integration as part of a pipeline that includes machine learning libraries such as sklearn [@pedregosa_scikit-learn_2011] and PyTorch[@paszkePyTorchImperativeStyle2019]. It is based on the principles of Configuration as Code, which enables the reproducibility of scientific ML experiments. To the best of our knowledge, no other packages comply with the flexibility and reproducibility of `ReciPies`. Existing libraries address parts of this workflow but leave important gaps that break the chain of reproducibility. Sklearn offers composable transformers, but no role-based variable grammar, limited human-readability, and awkward serialization beyond pickling. Packages like feature-engine [@galliFeatureenginePythonPackage2021], pyjanitor [@j.PyjanitorCleanerAPI2019], or scikit-lego[@warmerdamKoaningScikitlegoV0952025] add helpful transformers or cleaning verbs. However, none provide a unified, declarative recipe abstraction with a strict "prep/bake" split and backend flexibility. The R ecosystem's recipes package demonstrates the value of such a grammar, but no Python counterpart has consolidated these ideas. Given that Python is a dominant language for machine learning, `ReciPies` fills this gap by providing a stepwise recipe interface that is easy to use and read, allowing users to easily preprocess data for a wide range of machine learning pipelines.

# Usage
`ReciPies` is used as the bedrock of reproducible pipelines of Yet Another ICU Benchmark [@vandewaterAnotherICUBenchmark2024a], a flexible benchmarking framework for EHR and ICU models that has been adapted by the community in multiple works [@shenDataAdditionDilemma2024b; @santosImprovingRepresentationLearning2025]. The adaptable, configurable code modules that make extensive use of `ReciPies` can be found [here](https://github.com/rvandewater/YAIB/blob/development/icu_benchmarks/data/preprocessor.py). Note that `ReciPies` can be used for arbitrary research domains and is especially useful in domains where data is sequential.

We illustrate the use of `ReciPies` on an ICU mortality prediction task using the MIMIC‑IV database [@johnsonMIMICIVFreelyAccessible2023]. The dataset contains information about patients in the ICU, including vital signs, lab results, and medications. The pipeline requires timestamp-aware splits, role-based variable selection (IDs, outcomes, time), and consistent imputation/encoding across train–test partitions. With `ReciPies`, we declared these steps as a serializable recipe, *'prepped'* once on the training cohort, then *'baked'* on held‑out patients—eliminating data leakage and making each transformation auditable. We show how `ReciPies` can be used to preprocess this data and prepare it for machine learning tasks.

If we have a dataset, `df`, with a label `y`, some features `x1`, `x2`, `x3`, `x4`, an identifier `id`, and a sequential component `time`, we can build a preprocessing pipeline using `ReciPies`. We first do a train/test split:

``` Python
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
```
We then define the roles of the vriables in this dataset:
``` Python
roles = {outcomes:["y"], predictors=["x1", "x2", "x3", "x4"], groups=["id"], 
sequences=["time"]}
```
Now we can create the data `ingredients` which encapsulates the training data and its roles and the `recipe` that will be used to preprocess the data:
``` Python
ing = Ingredients(df_train, roles=roles)
rec = Recipe(ing)
```
We can now add preprocessing steps:
``` Python
rec.add_step(StepScale())
rec.add_step(StepSklearn(MissingIndicator(features="all"), 
  sel=has_role("predictor")))
rec.add_step(StepImputeFill(strategy="forward"))
rec.add_step(StepSklearn(LabelEncoder(), sel=has_type("categorical"), 
  columnwise=True))
```

We can now fit the recipe and transform both the train and test set without leakage:
``` Python
df_train = rec.prep()
df_test = rec.bake(df_test)
```
We can also use the `bake` method on the training set to transform it again, e.g., to apply the same transformations to a new dataset. Complete code, benchmarks, and interactive notebooks are available in the project documentation.

`ReciPies` also provides a benchmarking suite that allows users to compare the performance of different preprocessing steps on synthetic data. One can run on a variety of datasets and can be used to compare the performance of different preprocessing steps on a variety of sizes; datasets are getting increasingly bigger, and there is a real need for more performant pipelines. The benchmarks with Pandas and Polars backend on all steps and different data sizes are available in the repository and can be adapted to arbitrary datasets.

# Future steps
Several extensions could further improve `ReciPies`' utility and performance. We plan to expand the library of Polars-native steps to fully leverage its columnar execution model, particularly for time-series operations and large-scale aggregations where Polars shows significant performance advantages. Integration with MLOps frameworks, like Weights and Biases and MLFlow, represents another direction. We envision `ReciPies` recipes as portable preprocessing artifacts that can be versioned, tracked, and deployed across different environments. Tighter integration with experiment tracking and model registries would streamline the transition from research to production, a complex process in many application domains.

# Conclusion
`ReciPies` addresses a critical gap in the Python machine learning ecosystem by providing a declarative, reproducible framework for data preprocessing that prioritizes human readability and methodological transparency. Unlike existing solutions that either sacrifice readability for performance or flexibility for simplicity, `ReciPies` demonstrates that these trade-offs are unnecessary. Its role-based variable grammar, strict prep/bake separation, and backend flexibility create a preprocessing framework that scales from exploratory analysis to production deployment while maintaining full auditability; this is particularly valuable in regulated domains like healthcare and finance, where algorithmic decisions require justification and audit trails. By reducing the cognitive overhead of feature engineering and making methodological choices explicit, `ReciPies` enables researchers to focus on their core work. We hope this will broaden the reproducibility discussion in ML from just hyperparameters to the entire experiment pipeline. We encourage adoption, contribution, and extension by the machine learning community, particularly in developing domain-specific step libraries and integration patterns that can benefit the broader ecosystem.