# OISS Chatbot

This is a chatbot that allows users to ask immigration questions to their school's OISS website. The chatbot uses natural language processing to understand user questions and provide relevant answers. You can check it out here [OISS Chatbot](https://chat-with-oiss.streamlit.app/)

## Table of contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

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

- `API_KEY`: The API key for the OISS website.
- `API_URL`: The URL for the OISS website API.

You can also customize the HTML templates for the chatbot by editing the files in the `utils/htmlTemplates` directory.

## [Contributing](#contributing)

If you want to contribute to this project, you can fork the repository and submit a pull request with your changes. Please make sure to follow the coding style and conventions used in the existing code.

## [License](#license)

This project is licensed under the GNU AGPLv3 License. See the `LICENSE` file for more information.