# OISS Chatbot

This chatbot allows users to ask immigration questions to their school's OISS website. The chatbot uses natural language processing to understand user questions and provide relevant answers. You can check it out here [OISS Chatbot](https://chat-with-oiss.streamlit.app/)

## Table of contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- 
## [Overview](#overview)

### Project Overview

The key components of this project are:

1. **Data Ingestion and Indexing**: The codebase first ingests the content from the OISS website and creates a vector store using the Qdrant vector database. This allows for efficient retrieval of relevant information based on user queries.
2. **Chat Interface**: The application provides a Streamlit-based chat interface where users can enter their questions, and the chatbot responds with relevant information from the OISS website.
3. **Retrieval-Augmented Generation**: The project utilizes a Retrieval-Augmented Generation (RAG) approach, where the chatbot combines the retrieved information from the vector store with the capabilities of large language models to generate comprehensive and informative responses.
4. **API Key Management**: The application securely stores the user's API key for the selected chat model, allowing them to continue the conversation in future sessions.
5. **Prompt engineering:** The chatbot combines retrieved information with different prompts to fine-tune the responses generated by the LLM. ****I am still experimenting with this part, so you might not see any effects of prompt engineering on the deployed app.

### Code Overview

**src/app.py**: This is the main entry point of the Streamlit application.
- It handles the user interface, including the sidebar for selecting the university and chat model and the main chat interface.
- It manages the user's API key storage and retrieval using encryption.
- The 2 classes, **StreamHandler** and **PrintRetrievalHandler**, are responsible for handling the streaming output of the language model during the chat interaction and displaying the context retrieval process and the relevant documents during the chat interaction, respectively.
- It handles the chat input, passes it to the chain, and displays the chatbot's response.

**src/utils.py:** This file contains utility functions used across the application.
- It includes functions for encrypting and decrypting the user's API key and storing and retrieving the API key from the environment variables.
- It defines the chat model information, including the provider, API key link, and free trial status. It also contains the prompt template used for the Retrieval-Augmented Generation (RAG) approach.

**src/crud_collections.py**: This file handles the interactions with the Qdrant vector store. It has functions to create and manage Qdrant collections, which are used to store the OISS website content.

**src/utils/htmlTemplates.py**: This file contains the HTML templates used for the chat interface, including the CSS styles.

You might find some unnecessary files in the codebase due to a fixed cookie-cutter template I use for machine-learning projects. Ignore those.

## [Installation](#installation)

To install the dependencies for this app, you can create and activate a virtual environment using Conda and then run the following command:

```
pip install -r requirements.txt
```

Here are the steps to create and activate a virtual environment using Conda:

1. Install Conda if you haven't already. You can download and install Conda from the following link: https://docs.conda.io/en/latest/miniconda.html

2. Create a new virtual environment (feel free to change 'oiss-chatbot' to any name you prefer):

   ```
   conda create -n oiss-chatbot
   ```

3. Activate the virtual environment:

   ```
   conda activate oiss-chatbot
   ```

4. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

## [Usage](#usage)

To run the app, use the following commands:

```
cd src
streamlit run app.py
```

If you face any errors with langchain, try this command:
```
python -m streamlit run app.py
```

This will start the app and open it in a web browser. You can then ask immigration questions to your school's OISS website and receive answers from the chatbot.

## [Configuration](#configuration)

To configure the app, you can create a `.env` file in the root directory of the app and set the following environment variables:

- `QDRANT_HOST`: The host URL for the Qdrant database.
- `QDRANT_API_KEY`: The API key for accessing the Qdrant database.

If you plan to develop this app, you should create embeddings for the data using an embedding model you choose, store it in the vector database of your own choice, and provide the respective database's host URL and API key here. The code for fetching data and creating embeddings is present in the codebase. If you plan to use something other than Qdrant, you must work with respective APIs.

You can also customize the chatbots' HTML templates by editing the files in the `utils/htmlTemplates` directory.

## [Contributing](#contributing)

If you want to contribute to this project, you can fork the repository and submit a pull request with your changes. Please make sure to follow the coding style and conventions used in the existing code.

## [License](#license)

This project is licensed under the GNU AGPLv3 License. See the `LICENSE` file for more information.
