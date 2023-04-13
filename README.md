TITLE: 'Shining a light on ESG with NLP' 
==============================
**Important Note:**

Please note that for privacy reasons, the code base of this project is not hosted publicly. If you are interested in the code then please contact henry.paremain@gmail.com.

A summary video, report and slide deck are all available from this repo to demonstrate the output of the project.

**Abstract:**

This report investigates the application of natural language processing (NLP) techniques to streamline the ESG (Environmental, Social, and Governance) scoring process by summarising annual sustainability reports. The study leveraged a dataset of 1200 ESG reports web-scraped from responsibilityreports.com. LDA was the primary model developed, with secondary research exploring how ESGBert can be used for feature engineering. Ultimately, the LDA model was optimised using coherence scoring and trained on 1200 reports, leading to a novel ESG scoring method based on the frequency of topic mentions. The results of this study were integrated into an interactive Streamlit app, which employs the trained LDA model to generate customised insights based on the user's uploaded report.

In summary, this project demonstrates the potential of NLP techniques to facilitate the ESG scoring process, improve report summarisation, and offer valuable insights for investors and stakeholders.

**Walkthrough Demo of Streamlit App:**

https://www.veed.io/embed/88c8e1a3-e1d3-4a26-a109-6187c5d1a3e8

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
    ├── slide_deck.pdf           <- Non-technical slide deck of the project




