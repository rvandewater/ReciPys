# Concepts & Design Philosophy

`ReciPies` is a modular, declarative data preprocessing framework for Python, inspired by the "recipes" paradigm from R. It is designed to make machine learning pipelines **transparent**, **reproducible**, and **easy to audit**. Below, we outline the core concepts and the philosophy that guides the design of `ReciPies`.

______________________________________________________________________

## Core Concepts

### 1. **Ingredients**

- **Definition:** An `Ingredients` object wraps your dataset and attaches *roles* to each column (e.g., predictor, outcome, group, sequence).
- **Purpose:** This semantic labeling enables steps to operate on meaningful groups of variables, not just column names.
- **Example:**
    ```python
    roles = {"outcomes": ["y"], "predictors": ["x1", "x2", "x3", "x4"], "groups": ["id"], "sequences": ["time"]}
    ing = Ingredients(df, roles=roles)
    ```

### 2. **Recipe**

- **Definition:** A `Recipe` is an ordered list of *steps* that transform your data.
- **Purpose:** Encapsulates the entire preprocessing pipeline as a single, inspectable object.
- **Usage:** You add steps to a recipe, then `prep` (fit) it on training data and `bake` (apply) it to new data.
- **Example:**
    ```python
    rec = Recipe(ing)
    rec.add_step(StepScale())
    rec.add_step(StepImputeFill(strategy="forward"))
    rec.prep()  # Fit on training data
    rec.bake(df_test)  # Apply to test data
    ```

### 3. **Steps**

- **Definition:** Each *step* is a transformation (e.g., scaling, imputation, encoding) applied to selected columns.
- **Selection:** Steps use *selectors* to choose columns by role, type, or name.
- **Custom Steps:** Users can write their own steps for domain-specific logic.

### 4. **Selectors**

- **Definition:** Selectors are functions or objects that pick columns based on roles, types, names, or patterns.
- **Purpose:** Enables flexible, semantic selection of variables for each step.
- **Example:**
    ```python
    from recipies.selector import has_role, has_type

    rec.add_step(StepSklearn(LabelEncoder(), sel=has_type("categorical")))
    ```

### 5. **Prep & Bake**

- **Prep:** Fits all steps on the training data (learns statistics, encodings, etc.).
- **Bake:** Applies the fitted transformations to new data (test, validation, or production).
- **Guarantee:** Prevents data leakage by separating fitting and application.

______________________________________________________________________

## Design Philosophy

### **1. Configuration as Code**

- Pipelines are declared as code, not hidden in scripts or notebooks.
- Every transformation is explicit, inspectable, and versionable.

### **2. Human-Readability**

- Recipes are easy to read and audit.
- Variable roles make code self-documenting.

### **3. Reproducibility**

- Recipes can be serialized (JSON/YAML) and shared.
- Prep/bake split ensures transformations are reproducible and leakage-free.

### **4. Extensibility**

- Supports both Pandas and Polars backends.
- Users can add custom steps and selectors.

### **5. Transparency**

- Every step and selection is visible and inspectable.
- Provenance tracking is built-in.

______________________________________________________________________

## Why Use `ReciPies`?

- **Reduce cognitive load:** Focus on modeling, not boilerplate preprocessing.
- **Improve collaboration:** Share pipelines with clear intent and semantics.
- **Facilitate peer review:** Make methodological choices explicit.
- **Enable benchmarking:** Compare preprocessing strategies easily.

______________________________________________________________________

## Example Workflow

```python
from recipies import Ingredients, Recipe
from recipies.step import StepScale, StepImputeFill
from recipies.selector import has_role

roles = {"outcomes": ["target"], "predictors": ["age", "bp", "hr"], "groups": ["patient_id"], "sequences": ["timestamp"]}
ing = Ingredients(df, roles=roles)
rec = Recipe(ing)
rec.add_step(StepScale(sel=has_role("predictor")))
rec.add_step(StepImputeFill(strategy="forward"))
rec.prep()
df_test_transformed = rec.bake(df_test)
```

`ReciPies` brings clarity, reproducibility, and flexibility to your ML preprocessing pipelines.
