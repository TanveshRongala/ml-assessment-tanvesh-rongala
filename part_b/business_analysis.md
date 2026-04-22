# Part B: Business Case Analysis

## Promotion Effectiveness at a Fashion Retail Chain

---

# B1. Problem Formulation

## (a) Machine Learning Problem Formulation

**Target Variable**

The target variable is:

> Number of items sold per store per month under a given promotion

This represents the outcome the business wants to maximise.

---

**Candidate Input Features**

Possible predictive features include:

**Store-level features**

* Store size
* Monthly footfall
* Store location type (urban / semi-urban / rural)
* Local competition density
* Customer demographics

**Promotion-level features**

* Promotion type (Flat Discount, BOGO, Free Gift, Category Offer, Loyalty Points Bonus)
* Promotion duration
* Discount depth (if applicable)

**Time-based features**

* Month
* Season
* Festival indicator
* Weekend count in month

**Historical performance features**

* Past promotion effectiveness at store
* Previous month’s sales volume
* Rolling 3-month average sales

---

**Type of Machine Learning Problem**

This is a:

> Supervised regression problem

because:

* the target variable (items sold) is **continuous numeric**
* historical labelled data exists
* objective is predicting expected sales volume for each promotion option

The model can then support **decision optimisation** by selecting the promotion with the highest predicted output.

---

## (b) Why Items Sold Is Better Than Revenue as Target Variable

Using **sales revenue** alone can be misleading because:

Revenue depends on:

* product price differences
* discount depth
* promotion cost structure

For example:

A Flat Discount may increase revenue temporarily through high-priced items but sell fewer units than BOGO.

Since the business objective is:

> maximise number of items sold

sales **volume** is the correct optimisation target.

---

**Broader Principle Illustrated**

This reflects a key ML principle:

> The target variable must align directly with the business objective.

Choosing the wrong target variable leads to models that optimise the wrong behaviour even if predictions are accurate.

Correct target selection ensures:

* better decision-making
* meaningful recommendations
* measurable business value

---

## (c) Alternative to a Single Global Model

Instead of one global model across all stores, a better strategy is:

> Hierarchical (segmented) modelling

Examples include:

**Option 1: Cluster-based models**
Group stores into segments such as:

* urban stores
* semi-urban stores
* rural stores

Train separate models per segment.

**Option 2: Mixed-effects / multi-level model**
Capture both:

* global promotion behaviour
* store-specific variation

**Option 3: Store-level models with shared features**
Hybrid approach combining local and global signals.

---

**Justification**

Stores differ in:

* customer behaviour
* purchasing power
* competition
* promotion sensitivity

Segmented modelling improves:

* prediction accuracy
* promotion targeting
* interpretability

---

# B2. Data and EDA Strategy

## (a) Joining Tables and Dataset Grain

The four datasets are:

* transactions table
* store attributes table
* promotion details table
* calendar table

---

**Step 1: Aggregate Transactions Table**

Aggregate transactions to:

> store × month × promotion level

Example aggregations:

* total items sold
* number of transactions
* total revenue
* average basket size

---

**Step 2: Join Store Attributes**

Join using:

```
store_id
```

Adds:

* store size
* location type
* competition density
* demographics

---

**Step 3: Join Promotion Details**

Join using:

```
promotion_id
```

Adds:

* promotion type
* discount structure
* category restrictions

---

**Step 4: Join Calendar Data**

Join using:

```
month
```

Adds:

* festival indicator
* weekend count
* seasonality signals

---

**Final Dataset Grain**

Final modelling dataset:

> One row = one store × one month × one promotion

Each row represents expected promotion performance at that store during that month.

---

## (b) Exploratory Data Analysis (EDA)

Before modelling, the following analyses should be performed.

---

### 1. Promotion Type vs Items Sold (Boxplot)

Purpose:

Compare effectiveness of each promotion type.

Insights:

Identify:

* strongest performing promotions
* variability across promotions

Impact:

Guides feature encoding and baseline expectations.

---

### 2. Monthly Sales Trend (Time-Series Plot)

Purpose:

Detect seasonality patterns.

Example:

Higher sales during:

* festivals
* holiday seasons
* year-end periods

Impact:

Supports creation of seasonal features.

---

### 3. Store Location vs Promotion Effectiveness (Grouped Bar Chart)

Purpose:

Check whether promotion success differs by location type.

Example:

Urban stores may respond better to loyalty programs.

Impact:

Supports segmented modelling approach.

---

### 4. Correlation Heatmap

Purpose:

Identify relationships between numeric variables such as:

* footfall
* store size
* competition density
* items sold

Impact:

Helps:

* remove redundant variables
* select strong predictors
* detect multicollinearity

---

## (c) Impact of Promotion Imbalance (80% No Promotion)

Problem:

Dataset imbalance may cause the model to:

* under-learn promotion effects
* bias predictions toward non-promotion behaviour

Result:

Poor promotion recommendation accuracy.

---

**Solutions**

Possible mitigation steps:

1. Stratified sampling
2. Oversampling promotion months
3. Weighted loss functions
4. Separate promotion-effect model

These help the model properly learn promotion impact.

---

# B3. Model Evaluation and Deployment

## (a) Train-Test Split Strategy

Dataset:

* 3 years
* 50 stores
* monthly data

Correct split:

> Time-based split

Example:

Training set:

Year 1 + Year 2

Testing set:

Year 3

---

**Why Random Split Is Inappropriate**

Random splitting causes:

data leakage

because future information may enter training data.

Time-series problems must preserve:

chronological order

to simulate real deployment conditions.

---

**Evaluation Metrics**

Recommended metrics:

### 1. RMSE (Root Mean Square Error)

Measures:

average prediction error magnitude

Lower RMSE means:

better prediction accuracy.

---

### 2. MAE (Mean Absolute Error)

Measures:

average absolute difference between predicted and actual sales.

More interpretable than RMSE.

---

### 3. Promotion Recommendation Accuracy

Measures:

how often model selects best-performing promotion.

Most important business metric.

---

## (b) Explaining Different Monthly Recommendations Using Feature Importance

Example:

Model recommends:

December → Loyalty Points Bonus
March → Flat Discount

Explanation approach:

Use feature importance techniques such as:

* SHAP values
* permutation importance

---

Possible drivers:

**December**

Influenced by:

* festival season
* higher repeat customers
* gift shopping behaviour

Therefore:

loyalty promotions perform better.

---

**March**

Influenced by:

* lower seasonal demand
* price-sensitive shoppers

Therefore:

discount promotions perform better.

---

**Communication to Marketing Team**

Explain:

The model adapts recommendations based on:

* seasonality
* customer behaviour
* historical performance patterns

This increases confidence in model decisions.

---

## (c) Deployment Pipeline

Deployment involves three main stages.

---

### Step 1: Save the Model

Save trained model using:

```
pickle
joblib
MLflow
```

Store in:

model registry or cloud storage

---

### Step 2: Monthly Prediction Pipeline

Each month:

1. Collect latest store data
2. Merge with calendar features
3. Encode promotion options
4. Generate predictions for all promotions
5. Select promotion with highest predicted items sold

Output:

Recommended promotion per store.

---

### Step 3: Monitoring Model Performance

Track:

### 1. Prediction Error Over Time

Detect:

performance drift

---

### 2. Data Drift

Check changes in:

* customer behaviour
* store traffic
* promotion structure

---

### 3. Recommendation Success Rate

Compare:

predicted vs actual promotion performance

---

**Retraining Trigger**

Retrain model when:

* prediction error increases significantly
* promotion effectiveness changes
* new stores added
* new promotion types introduced

This ensures continued recommendation quality and business value.
