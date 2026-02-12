# ResolvePOI: Attribute Conflation

## Overview
Different data providers often describe the same real-world place (POI) with conflicting information. Even when two records refer to the same location, attributes like the name, category, phone number, website, or address may not match exactly.

The goal of this project is to study the problem of **place attribute conflation**: how to combine multiple POI records into a single, consistent representation by selecting the most accurate attribute values. This project focuses on analyzing how well different attributes align across sources and how confidence scores relate to correctness.

---

## Dataset
This project uses a dataset of **2,000 matched POI pairs**. Each row compares:
- a **candidate (conflated) POI record**, built from multiple data sources
- a corresponding **base POI record**, used as a reference for evaluation

Each POI includes structured attributes such as names, categories, websites, phone numbers, addresses, brand information, and source metadata. Many fields are stored as JSON objects to preserve primary values, alternates, and provenance.

The base POI should be treated as a **reference point**, not absolute ground truth. While it is generally higher quality, it may still contain errors, which reflects real-world data integration challenges.

---

## Data Structure
Each row contains identifiers (`id`, `base_id`), a set of aggregated attributes for the candidate POI, and corresponding base attributes prefixed with `base_`.

Examples of attributes include:
- `names` / `base_names`
- `categories` / `base_categories`
- `websites` / `base_websites`
- `phones` / `base_phones`
- `addresses` / `base_addresses`
- `confidence` / `base_confidence`

Most attributes include a primary value and, when available, alternate values.

---

## OKRs

### Objective 1: Create a reliable ground truth dataset for place attribute selection
The goal of this objective is to define what “correct” actually means when different sources disagree about a place, and to build a labeled dataset that can be used to evaluate any attribute selection logic.

- **KR1:** Manually label 1,500–2,000 pre-matched place pairs across core attributes such as name, phone number, website, address, and category to create a Golden Dataset.
- **KR2:** Reach at least 80% agreement on a shared subset of 200 labeled records, and explicitly document where and why disagreements occur instead of assuming there is always a single obvious answer.
- **KR3:** Write clear labeling guidelines for 5 key attributes, including how to handle formatting differences, missing values, and common edge cases.

---

### Objective 2: Build and test automated ways to resolve conflicting place attributes
This objective focuses on translating human decision-making into code by building systems that can automatically select the best attribute value when multiple versions disagree.

- **KR1:** Implement at least two attribute selection approaches (for example, a rule-based method and a simple ML-based method) and evaluate them using the Golden Dataset.
- **KR2:** Show that at least one approach performs significantly better than a simple baseline (such as always selecting the most recent or non-empty value) on one or more attributes.
- **KR3:** Analyze at least 15 failure cases per approach to understand where the system makes mistakes and which types of conflicts are hardest to resolve.

---

### Objective 3: Understand trade-offs and draw practical conclusions about conflation strategies
This objective focuses on stepping back from raw metrics to understand what actually works, what doesn’t, and why.

- **KR1:** Compare rule-based and ML-based approaches in terms of accuracy, interpretability, and complexity, and summarize the trade-offs observed.
- **KR2:** Identify 5 recurring edge cases where automated methods struggle, and suggest realistic ways these cases could be handled in practice.
- **KR3:** Write a final recommendation explaining which approach makes sense in different scenarios, such as large-scale processing versus higher-accuracy, lower-volume use cases.
