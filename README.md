TITLE: 'Shining a light on ESG with NLP' 
==============================

**Data:**

All pickles required for the notebooks can be found in the following google drive link:
https://drive.google.com/drive/folders/14Mgz1YHjDhcu6MNmcwIGRp2eZApT3_sG?usp=share_link

**Abstract:**

This report investigates the application of natural language processing (NLP) techniques to streamline the ESG (Environmental, Social, and Governance) scoring process by summarising annual sustainability reports. The study leveraged a dataset of 1200 ESG reports web-scraped from responsibilityreports.com. LDA was the primary model developed, with secondary research exploring how ESGBert can be used for feature engineering. Ultimately, the LDA model was optimised using coherence scoring and trained on 1200 reports, leading to a novel ESG scoring method based on the frequency of topic mentions. The results of this study were integrated into an interactive Streamlit app, which employs the trained LDA model to generate customised insights based on the user's uploaded report.

In summary, this project demonstrates the potential of NLP techniques to facilitate the ESG scoring process, improve report summarisation, and offer valuable insights for investors and stakeholders.

**Walkthrough Demo of Streamlit App:**

https://www.veed.io/embed/311d48be-ce2b-47cd-a543-f8633d2cf4b3

Project Flowchart
------------

![Alt text](/project_flowchart.svg)


Project Organization
------------

    ├── README.md               <- The top-level README for developers using this project.
    │
    ├── data                    <- All data files required to run notebooks (NB too large to upload to GIT - see
    │                           Google Drive link above to download data and place in this folder locally)
    │  
    ├── notebooks               <- Contains all final notebooks involved in the project
    │        
    ├── capstone_env.yml        <- The requirements file for reproducing the analysis environment
    │
    ├── project_flow_chart.svg  <- flow chart of project workflow
    │
    ├── final_report.pdf        <- written report which summarises the project
    │
    ├── slide_deck.pdf          <- Non-technical slide deck of the project




