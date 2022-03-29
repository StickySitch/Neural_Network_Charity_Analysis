# Neural_Network_Charity_Analysis
## Overview
Alphabet Soup has funded tens of thousands of applicant over the years. No one is ever looking to give money away for free, so to help Alphabet Soup improve their chances of funding successful operations, A deep learning model was developed using Python. 

Alphabet Soup has provided us with a dataset containing more than 34,000 organizations that received their funding. Using the features within the Alphabet Soup dataset, A binary classifier is created to predict if applicants would be successful if funded.

### `Dataset Columns:`
-   **`EIN`** and **`NAME`** — Identification columns
-   **`APPLICATION_TYPE`** — Alphabet Soup application type
-   **`AFFILIATION`** — Affiliated sector of industry
-   **`CLASSIFICATION`** — Government organization classification
-   **`USE_CASE`** — Use case for funding
-   **`ORGANIZATION`** — Organization type
-   **`STATUS`** — Active status
-   **`INCOME_AMT`** — Income classification
-   **`SPECIAL_CONSIDERATIONS`** — Special consideration for application
-   **`ASK_AMT`** — Funding amount requested
-   **`IS_SUCCESSFUL`** — Was the money used effectively
- 
## Results

### Preprocessing 

- **`What variable(s) are considered the target(s) for your model?`**
	- **`IS_SUCCESSFUL`:** This is our target because the `IS_SUCCESSFUL` tells us if they applicant was successful or not after funding.
	
- **`What variable(s) are considered to be the features for your model?`**
	- As you can see below, almost all the rest of the columns are considered features; All but two. The columns listed below are considered features due to their ability to add context to the model by showing what factors lead to success.

		-   `APPLICATION_TYPE` — Alphabet Soup application type
		-   `AFFILIATION` — Affiliated sector of industry
		-   `CLASSIFICATION` — Government organization classification
		-   `USE_CASE` — Use case for funding
		-   `ORGANIZATION` — Organization type
		-   `STATUS` — Active status
		-   `INCOME_AMT` — Income classification
		-   `SPECIAL_CONSIDERATIONS` — Special consideration for application
		-   `ASK_AMT` — Funding amount requested


- **`What variable(s) are neither targets nor features, and should be removed from the input data?`**
	- As mentioned above, there are two columns that are not features; But wait, they're also not the target? So what are they? Useless, really. In this use case they will dropped and unused.
		- **`EIN`** and **`NAME`** — Identification columns

### Compiling, Training, and Evaluating the Model
- **`How many neurons, layers, and activation functions did you select for your neural network model, and why?`**

The following number of neurons, layers, and activation functions were used for the neural network model:

-   **`First hidden layer`**
    -   Number of neurons: 80
    -   Activation function: `relu`
-   **`Second hidden layer`**
    -   Number of neurons: 30
    -   Activation function: `relu`
-   **`Third hidden layer`**
    -   Number of neurons: 20
    -   Activation function: `relu`
-  **` Ouput layer`**
    -   Activation function: `tanh`
- **`Total Neurons:`** 130 

As you can see, each hidden layer has the activation fuction "relu". One other layer, our `Output layer` has been swtich from `sigmoid` to `tanh`. This was done to see if we could improve our models accurracy score. There was a slight improvement of `.03%`.

- **`Were you able to achieve the target model performance?`**
	- No. Besides the slight increase of `.03%` from the added layer and changing `sigmoid` to `tanh`, I was not able to get anywhere close to a `75%` accuracy score.

- **`What steps did you take to try and increase model performance?`**
	- As mentioned above their were a few steps taken to try and improve the models accuracy:
		- **`Sigmoid to Tanh`:** With the trail and error approach in mind, I employed the power of the `Tanh` activation fuction to see if our models accuracy would improve. The output layer went from the `Sigmoid` activation to the `Tanh` activation, showing a slight increase in accuracy.
		- **`Third Layer`:** A third hidden layer was added to our model. To avoid overfitting, I wanted to keep the number of neurons low (20).  
		- **`Converted Dtypes`** Converted our `ASK_AMT` column to a `string`. The data within the column was binned by replacing values with less than 2,500 occurrences with `Other`. After encoding the `ASK_AMT` column with the other categorical variables, the accuracy score increased slightly.

## Summary

Well, all in all, it was a good effort but as you can see below, the highest accuracy score I was able to achieve was `66.5%`. Our goal was `75%` so let's talk about what we might change in order to achieve this. In my opinion, there is an easier and more efficient way to achieve this goal. It's a big change but I truly beleive that by scrapping the deep learning model and moving to a logistic regression model, we can achieve a much higher accuracy score. With all of the columns being encoded, the model would only need to determine the probability of belonging to one of two groups. 

![Model Output](https://github.com/StickySitch/Neural_Network_Charity_Analysis/blob/main/Resources/accuracy%20score.png)

