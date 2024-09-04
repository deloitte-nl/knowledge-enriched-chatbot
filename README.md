# **Knowledge-Enriched Chatbot**

# About

This project showcases a chatbot demo that utilizes Retrieval Augmented Generation (RAG) based on a Vector Database and a Knowledge Graph (KG) to enhance the accuracy and reliability of responses from Large Language Models (LLMs). You can test the chatbot with your own data or use the provided sample data about the greening initiatives of an imaginary city. The aim is to demonstrate how combining LLMs and RAG can result in more precise and source-based answers, making the chatbot a powerful tool for informed interactions.

### Technology Stack
- **Python**: Programming language used for development.
- **Neo4j**: Graph database for Knowledge Graphs.
- **Langchain**: Framework designed to simplify the creation of applications using large language models.
- **ChromaDB**: Vector store for document embeddings.

### Status
The project is currently a proof of concept and is not intended for production use.

### Future Directions
- **Enhanced Integration**: Further integration with other data sources.
- **Improved Performance**: Optimization of the retrieval and response generation process for a specific use case or data domain.
- **User Interface**: Development of a user-friendly interface for easier interaction.

# Installation

### Add your Azure OpenAI credentials
1. Replace the placeholders in `config.json` with Azure OpenAI credentials.
```json
{
   "OPENAI_API_KEY": "",
   "AZURE_OPENAI_ENDPOINT": ""
}
```
2. Replace the empty strings with your Azure OpenAI key, endpoint and version.

   *Note: The code currently uses 'gpt-35-turbo' and 'text-embedding-ada-002', so the API key should have both of these models deployed to work.*

### Python environment
Set up a Python environment and install the required packages by following these steps:
1. Create a New Conda Environment:
   ```bash
   conda create --name ke-chatbot python=3.11
   ```
2. Activate the Conda Environment:
   ```bash
   conda activate ke-chatbot
   ```
3. Install Required Packages:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This step may take a while as it installs all the necessary dependencies.*
4. Install the spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Install Neo4j
1. [Download Neo4j Desktop](https://neo4j.com/download/).
2. Run the installer
   - Execute the downloaded installer file.
   - Follow the on-screen prompts to complete the installation.
3. Start Neo4j Desktop and activate with the provided activation key.
4. Create a new database:
   - Add a new project
      -  Click the **'New'** button at the top left corner to create a new project.
   - Create the DBMS:
      - Within your new project, click the **'Add'** button at the top right corner.
      - Select **'Local DBMS'**.
      - Set a password for the database.
      - Choose version 5.1.0 from the version dropdown menu.
      - Click 'Create' to set up your local database instance.
5. Install plugins:
   - Click on the newly created DBMS to select it.
   - Go to the **'Plugins'** tab.
   - Install the following plugins:
      - **APOC:** A collection of procedures and functions for Neo4j.
      - **Neosemantics (n10s)**: A plugin for RDF and linked data.

6. Start and open the database.
   - Start your newly created database by clicking the **'Start'** button next to your database instance.
   - Once the database is running, click **'Open'** to access the Neo4j Browser interface.

   *Note: Conflicts may occur. If this happens, click on **'Fix configuration'***. If the problem persists, you may need to manually resolve it by editing the config file of the DBMS.*


7. Upload the ttl file that has the knowledge graph through the following code: 
   - If there is already another knowledge graph loaded to your graph database, start by running the following cypher code to clean it.
      ```bash
      // Delete nodes and relationships with expired TTL
      MATCH (n)
      DETACH DELETE n;
      ```
 
   - Now the knowledge graph which is stored in a tt file can be uploaded by running the below codes one by one. Change the path of the ttl file to the correct location before running it.
      ```bash
      // Create the necessary constraint 
      CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource) REQUIRE r.uri IS UNIQUE; 
      ```
      
      ```bash
      // Initialize or edit the Graph Config as needed  
      CALL n10s.graphconfig.init({handleRDFTypes:"LABELS_AND_NODES"}); 
      ```
      
      ```bash
      // Import RDF data from the specified file 
      CALL n10s.rdf.import.fetch("file:///**path to ttl file**", "Turtle", { format: "Turtle" });
      ```

8. Replace the placeholders in `config.json` with Neo4j Credentials:
```json
{
    "GRAPH_USERNAME": "Replace with your Neo4j URL",
    "GRAPH_PASSWORD": "Replace with your Neo4j username",
    "GRAPH_ENDPOINT": "Replace with your Neo4j password"
}
```

# Execution Flow

This section guides you through the process of running the main scripts in the project and understanding the sequence of operations.

### Main Scripts

1. **`main_vectordb_rag.py`**: The question-answering pipeline using RAG based on a vector store.
2. **`main_integrated_rag.py`**: The question-answering pipeline using RAG based on a vector store and knowledge graph combined. Make sure the graph in Neo4j is running while executing this script. 
3. **`evaluate.py`**: Evaluates the performance of the chatbot pipelines using predefined metrics.

### Step-by-Step Execution

#### `main_vectordb_rag.py`
1. **Data Loading**: Loads data from the specified Excel file using the `DocumentLoader` class.
2. **Document Chunking**: Chunks the loaded documents into smaller pieces using the `DocumentChunker` class.
3. **Vector Store**: Embeds the chunked documents and stores them in ChromaDB.
4. **QA Retrieval**: Uses the `CustomQARetriever` to perform question-answering by retrieving relevant documents from the vector store.


#### `main_integrated_rag.py`
1. **Data Processing**: Processes documents by loading, chunking, and embedding them (if `load_from_existing_vector_db` is set to False).
2. **Cypher QA Retrieval**: Uses the `CypherQARetriever` to generate knowledge graph context by translating user questions into Cypher queries.
3. **Vector Store Retrieval**: Retrieves relevant documents from the vector store using the `CustomQARetriever`.
4. **Integrated QA**: Combines the context from both the knowledge graph and vector store to provide a comprehensive answer.

#### `evaluate.py`
1. **Initialize Evaluation Arguments**: Sets up parameters for evaluation.
2. **Load Data and Configuration**: Initializes components like the QA retriever and vector store.
3. **Run Evaluation Pipeline**: Executes the evaluation flow, which includes generating synthetic datasets, running the RAG module, and evaluating the answers.
4. **Save Results**: Outputs the evaluation results, including aggregated metrics and datasets.

### Parameters and Configuration
Key parameters and configurations can be set in the `config.py` file.

### Example Commands
- Run the question-answering pipeline with vectorDB supported RAG
   ```bash
   python main_vectordb_rag.py
   ```
- Run the question-answering pipeline with vectorDB and KG supported RAG
   ```bash
   python main_integrated_rag.py
   ```
- Run the evaluation script
   ```bash
   python evaluate.py
   ```

### Logging and Output
   - Logs and output results are saved in the specified directories.
   - Check the console output for detailed logging information.

### Troubleshooting
- **Common Issue 1**: If you encounter an error with ChromaDB, ensure the database path is correctly set.
- **Common Issue 2**: If the Neo4j database fails to start, verify that the correct version is installed and running.



# Running the integrated RAG pipeline with your own data

To run the integrated pipeline, the Knowledge Graph requires a TTL data file of your data. This section guides you through creation of TTL file from CSV data. If you want to create a knowledge graph from tabular data such as CSV files, follow these step-by-step instructions. We'll use a simple example to illustrate the process.

### Example CSV Data

Here is a sample CSV file containing information about cities and their number of citizens:

```bash
ID_city,City,Number_of_citizens
1,Amsterdam,123
2,Berlin,234
```

### Conceptual Model

The conceptual model for this example will have the following entity:

- **City**: This entity will have a property for the number of citizens.

### TTL Output Example

To convert the above CSV data into RDF (TTL format), you can use the following template and script.

1. **Conceptual Model in TTL format**:

   ```bash
   @prefix ex:  .
   @prefix rdfs:  .

   ex:City_1 a ex:City ;
      rdfs:label "Amsterdam" ;
      ex:Number_of_citizens "123" .

   ex:City_2 a ex:City ;
      rdfs:label "Berlin" ;
      ex:Number_of_citizens "234" .
   ```

2. **Python Script for Conversion**: You can use the following Python script to automate the conversion of CSV data to RDF in TTL format.

   ```python
   import csv

   # Define prefixes and namespaces
   ttl_prefixes = """
   @prefix ex:  .
   @prefix rdfs:  .
   """

   # Function to generate TTL content from CSV rows
   def generate_ttl(csv_file):
      ttl_content = ttl_prefixes + "\n"
      with open(csv_file, mode='r') as file:
         csv_reader = csv.DictReader(file)
         for row in csv_reader:
               ttl_content += f"""
   ex:City_{row['ID_city']} a ex:City ;
      rdfs:label "{row['City']}" ;
      ex:Number_of_citizens "{row['Number_of_citizens']}" .
   """
      return ttl_content

   # Path to your CSV file
   csv_file_path = 'cities.csv'
   ttl_content = generate_ttl(csv_file_path)

   # Write TTL content to a file
   with open('cities.ttl', 'w') as ttl_file:
      ttl_file.write(ttl_content)
   ```

3. **Steps to Run the Script**:

   - Save your CSV data into a file named cities.csv.
   - Save the Python script above into a file named convert_csv_to_ttl.py.
   - Run the script:
      ```bash
      python convert_csv_to_ttl.py
      ```
   This will generate a file named cities.ttl containing the RDF representation of your CSV data.

4. **Upload the TTL file to Neo4j**:

- Use the instructions provided in the Neo4j installation section to upload the generated ttl file to your Neo4j database.

By following these steps, you can convert your tabular data from CSV files into a knowledge graph in TTL format, enabling the integration with Neo4j and other RDF-based systems.



