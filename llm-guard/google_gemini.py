# Install Vertex AI
# pip install google-cloud-aiplatform --upgrade --user

# Install LLM guard reference https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_python.ipynb for more detailed instructions
# pip install llm-guard

# Set Project and Location
PROJECT_ID = "YOUR PROJECT ID"  # @param {type:"string"}
LOCATION = "THE REGION"  # @param {type:"string"}

# Authenticate yourself (remove google.colab if not using Colab)
# If you are using Vertex AI Workbench, check out the setup instructions here:
# https://github.com/GoogleCloudPlatform/generative-ai/tree/main/setup-env

# Set the Model in this example we are using Gemini Pro
from vertexai.preview.generative_models import GenerativeModel

# Load Gemini Pro
generation_model = GenerativeModel("gemini-2.0-flash")


# Import Scanners from LLM Guard
from llm_guard import scan_output, scan_prompt
from llm_guard.input_scanners import PromptInjection, TokenLimit, Toxicity
from llm_guard.output_scanners import NoRefusal, Relevance, Sensitive

# Give a prompt
prompt = "What is Sundar's email?"

# Generate a response using Gemini Pro
response = generation_model.predict(prompt)

# Set your values for LLM Guards Input Scanners, this example uses the defaults
input_scanners = [TokenLimit(), Toxicity(), PromptInjection()]

# Set response variables
sanitized_prompt, results_valid, results_score = scan_prompt(input_scanners, prompt)

# Set Values for LLM Guards Output Scanners, this example uses the defaults
output_scanners = [NoRefusal(), Relevance(), Sensitive()]

# Set the Variables for Output scanners results
sanitized_response_text, output_results_valid, output_results_score = scan_output(
    output_scanners, prompt, response.text, False
)

# Run a test and use the scanners
# Run the Input Scanner
scan_prompt(input_scanners, prompt)
# Run the output Scanner
scan_output(output_scanners, prompt, response.text)

# Check if the prompt and response are valid
if all(results_valid.values()) is True:
    if not all(output_results_valid.values()) is True:
        print(f"Prompt: {prompt} is not valid")
    else:
        print(response.text)
else:
    print("Prompt is not valid")
