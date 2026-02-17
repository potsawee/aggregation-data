# Social Attitudes and Values Survey Dataset 

This dataset contains survey questions and responses designed to explore social attitudes and values among people in Thailand in 2025. It includes a comprehensive set of carefully crafted questions and collected responses aimed at facilitating research on social perspectives, values, cultural attitudes, as well as crowdsourcing algorithm research. This dataset was used to evaluate our proposed crowdsourcing algorithm. 

## Dataset Structure

The dataset contains 1,000 examples (with 20 questions repeated for quality control). The data was constructed in English (using contexts relevant to the Thai culture) and then translated into Thai for Thai respondents. There are two splits:

- English (en): 1,020 examples 
- Thai (th): 1,020 examples (the translated version which was presented to respondents)

Note that the column `responses` are identical in both splits. 

### Features:

- `Q_ID` (string): Unique identifier for each question.
- `context` (string): Contextual information provided for each question.
- `question` (string): The survey question.
- `scale` (string): Descriptions of the -5, 0, +5 scale points on an 11-point scale.
- `responses` (list of integers): Collected responses from 39 workers* (in the same order from `respondent1`, ..., `respondent39`).
- `repeat` (string): Indicates if a question was repeated for quality control purposes. If the value is not `None`, the value is `Q_ID` of the original question.

*Originally, 40 workers were recruited but 1 worker did not follow the instruction, hence, their responses are not included in this dataset

## Dataset Creation Process

### Question Generation

- **Topic Selection**: Initial topics (420 in total) were sourced from the Pew Research Center. Topics were evaluated by the Claude 3.5 Sonnet model based on clarity, general knowledge required, and sensitivity. Topics with ratings ≥ 3 (out of 5) were retained, resulting in 301 topics.

- **Question Generation**: 10 survey questions were generated per topic using Claude 3.5 Sonnet, yielding 3,010 initial questions. Each data point includes context, a question, and descriptions for scores of -5, 0, and +5.

- **Question Refinement**: Questions underwent further evaluation for clarity, relevance, accuracy, and neutrality of scale descriptions, narrowing the final set down to 1,000 unique questions.

### Data Annotation and Collection

- **Worker Annotation**: Responses were collected from 40 recruited Thai annotators, predominantly students and freelancers. Annotators responded to 1,020 questions (1,000 unique + 20 repeated) presented in Thai via Google Forms.

- **Quality Control**:
  - Workers were instructed to limit zero-score responses to under 10%.
  - Repeated questions were included to assess response consistency.
  - Data from annotators failing quality guidelines was excluded.

## Usage and Applications

The dataset is suitable for research in:

- Crowdsourcing Algorithm
- Statistical Analysis for Cross-cultural Studies, Opinion mining, etc

## Dataset Statistics

- **Total Size**: ~1.7 MB
- **Download Size**: ~420 KB
- **Number of Examples**: 1,020
- **Numer of Respondents**: 39

## Licensing

This dataset is licensed under a [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

You are free to share, adapt, and use this dataset for any purpose, even commercially, provided that appropriate credit is given.