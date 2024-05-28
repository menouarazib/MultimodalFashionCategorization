Enhancing Classification with Multimodality: A Hierarchical Approach to Fashion Item Categorization
==========================================================
## Accepted for FTC 2024 - Future Technologies Conference 2024


<h1 align="center">
<img src="https://raw.githubusercontent.com/menouarazib//MultimodalFashionCategorization/444f122fbd0c718d543e306304ba308329879cdd/images/architcture.jpg" width="800">
</h1><br>

**Abstract**

In the dynamic e-commerce landscape, online sellers face the time-consuming task of classifying and categorizing items into appropriate categories. This process is crucial for ensuring product information accuracy and safeguarding consumer interests. Traditionally, manually creating item images and descriptions has been the preferred method to avoid potential inaccuracies and detrimental consequences that could arise from AI-generated errors. However, the advent of deep learning has opened up new avenues for automatic classification tasks. This paper explores the concept of multimodality, a technique that combines an item's image and description, to achieve high accuracy in hierarchical item categorization. We present a comprehensive analysis of its effectiveness on fashion items and demonstrate how it significantly enhances automatic categorization compared to other approaches that do not consider multimodality.

**Keywords**: Multimodality, Computer Vision, Natural Language Processing, Hierarchical Categorization, Fashion Items, E-commerce Product Classification.

| Method | SMOTE | Multimodality | Category Level 1 | Category Level 2 | Category Level 3 |
| --- | --- | --- | --- | --- | --- |
| Fengzi Li *et al.* [li2020neural] | Yes | No | N/A | 0.957 | N/A |
| Brendan Kolisnik *et al.* [kolisnik2021condition] | No | No | 0.997 | 0.980 | 0.910 |
| Ours (Configuration Base) | No | Yes | 0.997 | 0.985 | 0.960 |
| Ours (Configuration Large) | No | Yes | **0.999** | **0.999** | **0.9995** |


**Table 1:** Comparison of validation accuracy across different levels for hierarchical category classification.
