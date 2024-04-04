---
layout: page
title: Projects
permalink: /projects/
---
<!-- [go to heading](#Skills) -->


Explore some of the projects I've worked on below. Click on a project title to jump to its detailed description.

#### 2024 🐲

#### 2023 🐰
- [Music lyrics Similaries match](#system-for-automatic-data-extraction-and-storage)

#### 2022 🐯
- [System for Automatic Data Extraction and Storage](#system-for-automatic-data-extraction-and-storage)
- [An Interactive Response System Based on Intent Recognition](#an-interactive-response-system-based-on-intent-recognition)

#### 2021 🐮
- [Product Profile and Sentiment Analysis](#product-profile-and-sentiment-analysis)

#### 2019 🐷
- [Classifying Deep Features in an Urban Environment](#classifying-deep-features-in-an-urban-environment)
- [Mercari Price Suggestion Challenge - Kaggle Competition](#mercari-price-suggestion-challenge-kaggle-competition)
- [Learning to Play Space Invaders Using Deep Q-Networks](#learning-to-play-space-invaders-using-deep-q-networks)
- [Evaluation of Bayesian Modelling Methods on Energy Efficiency Dataset](#evaluation-of-bayesian-modelling-methods-on-energy-efficiency-dataset)
- [Electric Vehicles Prediction](#electric-vehicles-prediction)

#### 2015 🐑
- [Monte Carlo Method and Its Applications](#monte-carlo-method-and-its-applications)
- [Fractal Jet Nozzle Array - Patent for Invention](#fractal-jet-nozzle-array-patent-for-invention)

---


## System for Automatic Data Extraction and Storage
{: #system-for-automatic-data-extraction-and-storage }

- **Overview**: This project aimed to design an automatic data processing pipeline for an insurance company that could extract and store unstructured data into structured data templates.

- **Approach**: The system utilized Large Language Models (LLMs) to recognize and extract unstructured data into a data pool. The extracted data was then fulfilled into format templates, enabling business users or agents to access and utilize the structured data.

- **Pipeline**:
  - Design structured data templates and data types for regular insurance business data
  - Use LLMs to recognize and extract unstructured data into a data pool
  - Fulfill the extracted data into the format templates
  - Provide a feedback loop for enriching and correcting the diversity of data

- **Benefits**: The system aimed to decrease data engineering time and costs by automating the data processing pipeline, enabling efficient storage and utilization of unstructured data.

## An Interactive Response System Based on Intent Recognition
{: #an-interactive-response-system-based-on-intent-recognition }

- **Overview**: This project focused on developing a chatbot system for an insurance company that could recognize user intentions and automatically complete business multi-tasks.

- **Objectives**: The system aimed to provide efficient and effective insurance customer service by leveraging advanced natural language processing techniques.

- **Modules**:
  - Intention recognition: Using BERT-based and GPT-based models to recognize user intentions
  - Task assignment: Automatically assigning tasks to appropriate domain models based on user intentions
  - Domain models: Specialized models for handling policy inquiries, claims processing, billing inquiries, etc.
  - Output generation: Generating text, images, and links to provide necessary information to users

- **Approach**: The project involved research and development, including training BERT-based and GPT-based models for intent recognition, developing task assignment modules, and building domain-specific models for various insurance-related tasks.

- **Benefits**: The chatbot system aimed to improve customer service by providing 24/7 assistance, reducing wait times, and increasing customer satisfaction, while also reducing costs by minimizing the need for human customer service representatives.


---

## Product Profile and Sentiment Analysis
{: #product-profile-and-sentiment-analysis }

[Download Product Profile Technical Report (PPT)](/files/product_profile.pdf)
- **Overview**: This project focused on developing and implementing methods for sentiment analysis and labeling system on product comments from TianMao, Taobao and Jingdong, e-commerce platforms in China.

- **Objectives**: The primary goals were to help companies better understand customer needs, address current or predictable problems, and increase profits through effective sentiment analysis and multi-label classification.

- **Approach**: Several implementable methods for sentiment analysis on product comments were proposed and evaluated for their effectiveness and efficiency.

- **DITING Tool**: The project involved developing an online analysis tool called DITING, which aimed to become a new business intelligence software capable of:
  - Comprehensive analysis of pre-sale, post-sale conversations, and product comments
  - Identifying key factors in the purchase process
  - Providing objective analysis results by incorporating features like consultation rate, conversion rate, and refund rate
  - Generating complete solutions, including consumer insights and implementation methods

- **Methodology**: A language model was pre-trained and fine-tuned to complete sentiment analysis and multi-classification tasks. CBOW was employed for word embedding, and LSTM and GRU were used to build models. This approach proved effective for sentiment analysis in the e-commerce industry, achieving high accuracy.

---

## Classifying Deep Features in an Urban Environment
{: #classifying-deep-features-in-an-urban-environment }
### Master's Thesis. University of Bath.

[Download M (PDF)](link-to-your-cv.pdf)
- **Overview**: This project aimed to implement an end-to-end method for urban classification based on deep features extracted from satellite images. The objective was to develop an easy-to-use technique for road pattern extraction that could capture the character of urban layouts and categorize different cities around the world.

- **Approach**: Deep learning architectures, specifically Convolutional Neural Networks (CNNs), were employed to simulate the human brain's pattern recognition capabilities. The project focused on finding a way to achieve high accuracy in a short time by studying the performance of various hyperparameters and activation functions.

- **Results**: The CNN algorithm successfully completed urban classification for 10 cities using a dataset of 1km images. The model achieved an impressive accuracy of 93.2% after 2,800 epochs of training.

- **Insights**: This project demonstrated the potential of deep learning techniques for urban classification tasks and provided insights into the impact of hyperparameters and activation functions on model convergence rates.

## Mercari Price Suggestion Challenge - Kaggle Competition
{: #mercari-price-suggestion-challenge-kaggle-competition }

- **Overview**: This project involved participating in the Mercari Price Suggestion Challenge hosted on Kaggle, a renowned platform for data science competitions. The challenge required analyzing a dataset of product listings to build a machine learning model capable of suggesting accurate prices for new product listings.

- **Approach**: collaborated with a team and followed a structured approach, including data engineering, exploratory data analysis, algorithm building, and model evaluation and comparison. Various machine learning algorithms, such as regression models, tree-based methods, and ensemble techniques, were experimented with to find the best-performing model for price prediction.

- **Results**: The team achieved an impressive 80% accuracy in predicting product prices on the test set, showcasing the ability to tackle real-world challenges and deliver accurate results.

- **Insights**: This project allowed for hands-on experience in data engineering, exploratory analysis, algorithm development, and model evaluation within a competitive setting, further enhancing machine learning skills.


## Learning to Play Space Invaders Using Deep Q-Networks
{: #learning-to-play-space-invaders-using-deep-q-networks }

[Download Deep Q-Networks Technical Report (PDF)](/files/RL_Project_Report.pdf))
- **Overview**: This project was been applied deep reinforcement learning techniques, specifically Deep Q-Networks (DQN), to the classic Atari 2600 game, Space Invaders. The goal was to develop an artificial intelligence agent capable of playing the game by leveraging deep neural networks and reinforcement learning algorithms.

- **Approach**: To handle the high-dimensional state space represented by game screen images, developed a Convolutional Neural Network (CNN) architecture that could effectively process the game screens and provide an optimized Q-policy for selecting actions. I experimented with various hyperparameters, such as the experience replay batch size and training duration, to fine-tune the model's performance.

- **Results**: After extensive testing and validation, the best model trained was a DQN with preprocessed image frames and an experience replay batch size of 16, trained over the course of one day. This model demonstrated impressive performance in playing the Space Invaders game.

- **Insights**: Throughout this project, gained a deeper understanding of deep reinforcement learning and explored advanced techniques like Double DQN, Dueling DQN, and Inverse Reinforcement Learning, which provided valuable insights into the latest developments in the field.

## Evaluation of Bayesian Modelling Methods on Energy Efficiency Dataset
{: #evaluation-of-bayesian-modelling-methods-on-energy-efficiency-dataset }

- **Overview**: This project focused on evaluating Bayesian modeling methods for a multivariate regression task using an "energy efficiency" dataset. The objectives were to derive a good predictor for the data and estimate which input variables were relevant for prediction.

- **Approach**: The project involved approximating posterior distributions using the Hamiltonian Monte Carlo (HMC) stochastic method. The data set contained 768 examples with eight input variables representing basic architectural parameters for buildings, with the goal of predicting a ninth variable, the required "Heating Load".

- **Methodology**: The project was divided into several sub-tasks, including:
  - Exploratory analysis of the dataset
  - Applying standard Bayesian linear regression models
  - Using the Hamiltonian Monte Carlo algorithm
  - Estimating relevant input variables
  - Experimenting with new features
  - Modifying the HMC sampling framework for classification

- **Bayesian Linear Regression**: The posterior distribution was visualized over a grid of log(alpha) and log(sigma^2), and the maximizing values were used to compute the posterior mean weights. The RMS error was then calculated for both training and test sets.

- **Hamiltonian Monte Carlo (HMC) Sampling Algorithm**: The accuracy of the sampler depended on the simulation parameters L and epsilon0. The HMC sampler was tested on a correlated Gaussian, and the results were compared to samples taken from the known Gaussian.

- **Automatic Relevance Determination (ARD)**: The project explored incorporating ARD into the model, which involved using a Gaussian prior with one precision parameter for each weight.

- **Results**: The project provided insights into the application of Bayesian methods for regression tasks, including the visualization of posterior distributions, the use of HMC for sampling, and the incorporation of ARD for feature selection.

## Electric Vehicles Prediction
{: #electric-vehicles-prediction }

- **Overview**: This project involved analyzing data from the Department for Transport to predict the condition of the traffic network and the adoption of electric vehicles in the United Kingdom.

- **Data Sources**:
  - Road level Annual Average Daily Flow (AADF) estimates (2018) - major road by direction
  - Road level traffic volume estimates, Traffic-major roads (miles) (2018)

- **Approach**: The project utilized data from various sources, including local authorities, local governments, and the Ordnance Survey. The data included information on road lengths, vehicle counts, and traffic volumes.

- **Objectives**: The project aimed to analyze the collected data to gain insights into the current state of the traffic network and make predictions about the adoption of electric vehicles in the UK.

---

## Monte Carlo Method and Its Applications
{: #monte-carlo-method-and-its-applications }

- **Overview**: This research explored the application of the Monte Carlo method for European option pricing in the financial domain. The project aimed to predict option prices based on the volatility of underlying assets using stochastic simulations.

- **Approach**: The Monte Carlo method was utilized to simulate random paths of stock prices, following specific rules and random variables. These simulated stock prices were then used to calculate the revenue of European options based on the Black-Scholes model and binary tree model.

- **Results**: The project successfully implemented the Monte Carlo method for European option pricing, allowing for accurate predictions by accounting for the volatility and uncertainty in financial markets.

- **Insights**: This interdisciplinary research demonstrated the applicability of physics-based methods, such as the Monte Carlo method, in the financial domain, showcasing the potential for cross-disciplinary approaches in solving complex problems.

## Fractal Jet Nozzle Array - Patent for Invention
{: #fractal-jet-nozzle-array-patent-for-invention }
####  Patent Number ZL201310616228.9, China.

- **Overview**: In collaboration with five colleagues, worked on designing a fractal jet nozzle array, which resulted in a patent for invention. This project involved constructing the parameters of the jet model and conducting simulation tests to optimize the design.

- **Approach**: Began by designing the initial jet model using computational fluid dynamics (CFD) simulations. Then, iteratively adjusted the parameters, such as nozzle geometry and flow conditions, to improve the performance of the jet array. To facilitate this process, utilized the Orange software, a powerful data mining and machine learning toolkit, to calculate and analyze the simulation results.

- **Results**: Through a series of simulations and parameter adjustments, I were able to optimize the design of the fractal jet nozzle array, ensuring efficient mixing and desired flow characteristics.

- **Insights**: This project showcased my ability to work in a collaborative environment, apply computational techniques to physical problems, and iterate on designs to achieve optimal solutions, demonstrating versatility and problem-solving skills.



Go to top tables [⬆️](/projects/)
