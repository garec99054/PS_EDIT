import os
import json
import logging
from typing import List, Dict, Tuple, Any
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time
from anthropic import InternalServerError, RateLimitError, APIStatusError
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAIN_PS_FILE = "ARTEMIS/DOC_FILES/main_ps_draft_v1.txt"
COMMENTS_FILE = "ARTEMIS/DOC_FILES/COMMENTS/comments.txt"
TASKS_FILE = "ARTEMIS/GENERATED/TASKS/personal_statement_tasks.txt"
CONTENT_JSON_FILE = "ARTEMIS/GENERATED/CONTENT_FOR_PS/CONTENT_DOCS/personal_statement_content.json"
SUPPORTING_DOCS_INDEX_PATH = "ARTEMIS/INDEXED/SUPPORTING_INDEXED"
PRIORITIZED_EXCERPTS_INDEX_PATH = "ARTEMIS/INDEXED/NOTES/prioritized_excerpts"
RANDOM_PS_NOTES_INDEX_PATH = "ARTEMIS/INDEXED/NOTES/random_ps_notes"
REF_PS_INDEX_PATH = "ARTEMIS/INDEXED/INDX_PS"
REF_PS_ANALYSIS_FILE = "ARTEMIS/DOC_FILES/REF_PS_ANALYSIS_JSON/reference_ps_analysis.json"
ANALYZED_INDEX_PATH = "ARTEMIS/INDEXED/ANALYZED_INDEX"
CONTENT_INDEX_PATH = "ARTEMIS/GENERATED/CONTENT_FOR_PS/CONTENT_INDEXED"

OUTPUT_FOLDER = "ARTEMIS/DRAFTS"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "claude-3-5-sonnet-20240620"

def retry_on_error(max_retries=5, delay=5, initial_delay=1, max_delay=60):
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_delay = delay or initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (InternalServerError, RateLimitError, APIStatusError) as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Error occurred: {str(e)}. Attempt {attempt + 1}/{max_retries}. Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    if not delay:  # Only use exponential backoff if delay is not specified
                        current_delay = min(current_delay * 2, max_delay)  # Exponential backoff with max delay
            return func(*args, **kwargs)
        return wrapper
    return decorator

def load_index(persist_dir: str) -> VectorStoreIndex:
    """Load an indexed document."""
    logger.info(f"Loading index from {persist_dir}")
    try:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
        Settings.llm = Anthropic(model=LLM_MODEL)
        index = load_index_from_storage(storage_context)
        logger.info("Index loaded successfully")
        return index
    except Exception as e:
        logger.error(f"Failed to load index: {e}", exc_info=True)
        raise

def load_main_ps(file_path: str) -> str:
    """Load the main personal statement from a text file."""
    logger.info(f"Loading main personal statement from {file_path}")
    try:
        with open(file_path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load main personal statement: {e}", exc_info=True)
        raise

def load_text_file(file_path: str) -> str:
    """Load text from a file."""
    logger.info(f"Loading text from {file_path}")
    try:
        with open(file_path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load text file: {e}", exc_info=True)
        return ""

def load_json_file(file_path: str) -> List[Dict]:
    """Load JSON data from a file."""
    logger.info(f"Loading JSON from {file_path}")
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON file: {e}", exc_info=True)
        return []

def load_reference_ps_analysis(file_path: str) -> Dict:
    """Load the reference personal statement analysis from JSON file."""
    logger.info(f"Loading reference PS analysis from {file_path}")
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load reference PS analysis: {e}", exc_info=True)
        return {}

import time
from anthropic import InternalServerError, RateLimitError

def search_content(index: VectorStoreIndex, query: str, max_retries: int = 3, retry_delay: float = 1.0) -> str:
    """Search for content in the given index with retry logic."""
    for attempt in range(max_retries):
        try:
            query_engine = index.as_query_engine()
            response = query_engine.query(query)
            return str(response)
        except (InternalServerError, RateLimitError) as e:
            if attempt < max_retries - 1:
                logger.warning(f"API error occurred: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"Failed to search content after {max_retries} attempts: {e}", exc_info=True)
                return f"Error: Unable to retrieve content after {max_retries} attempts."
        except Exception as e:
            logger.error(f"Unexpected error occurred while searching content: {e}", exc_info=True)
            return f"Error: An unexpected error occurred: {str(e)}"

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate the cosine similarity between two texts."""
    stop_words = set(stopwords.words('english'))
    
    def preprocess(text):
        tokens = word_tokenize(text.lower())
        return ' '.join([word for word in tokens if word.isalnum() and word not in stop_words])
    
    preprocessed_text1 = preprocess(text1)
    preprocessed_text2 = preprocess(text2)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([preprocessed_text1, preprocessed_text2])
    
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def ensure_original_content(original_ps: str, modified_ps: str) -> str:
    """Ensure the modified personal statement is based on the original content."""
    llm = Anthropic(model=LLM_MODEL)
    
    prompt = f"""
    Original Personal Statement:
    {original_ps}

    Modified Personal Statement:
    {modified_ps}

    Your task is to review the modified personal statement and ensure it is primarily based on the original statement. 
    Make adjustments to the modified statement to:
    1. Preserve the core content and experiences from the original statement.
    2. Maintain the personal voice and unique aspects of the original.
    3. Incorporate improvements and refinements from the modified version.
    4. Remove any content that doesn't align with the original experiences.
    5. Ensure the final statement is coherent, well-structured, and within 800 words.

    Please provide the final version of the personal statement that closely aligns with the original while incorporating valuable improvements:
    """

    try:
        response = llm.complete(prompt)
        final_ps = response.text.strip()
        return final_ps
    except Exception as e:
        logger.error(f"Failed to ensure original content: {e}", exc_info=True)
        return modified_ps  # If the process fails, return the modified version

def is_personal_statement_complete(ps: str) -> bool:
    """Check if the personal statement is complete."""
    # Check if the statement has a clear introduction, body, and conclusion
    paragraphs = ps.split('\n\n')
    if len(paragraphs) < 3:
        return False
    
    # Check if the word count is within the expected range (500-800 words)
    word_count = len(ps.split())
    if word_count < 500 or word_count > 800:
        return False
    
    # Check if it covers key elements (motivation, experiences, future goals)
    key_elements = ['motivation', 'experience', 'goal']
    if not all(element in ps.lower() for element in key_elements):
        return False
    
    return True

class PersonalStatementAgent:
    def __init__(self):
        self.llm = Anthropic(model=LLM_MODEL, max_tokens=4000)
        self.main_ps = ""
        self.comments = ""
        self.tasks = ""
        self.content = []
        self.supporting_docs_index = None
        self.prioritized_excerpts_index = None
        self.random_ps_notes_index = None
        self.ref_ps_index = None
        self.analyzed_index = None
        self.content_index = None
        self.ref_ps_analysis = {}

    def load_data(self):
        self.main_ps = load_main_ps(MAIN_PS_FILE)
        self.comments = load_text_file(COMMENTS_FILE)
        self.tasks = load_text_file(TASKS_FILE)
        self.content = load_json_file(CONTENT_JSON_FILE)
        self.supporting_docs_index = load_index(SUPPORTING_DOCS_INDEX_PATH)
        self.prioritized_excerpts_index = load_index(PRIORITIZED_EXCERPTS_INDEX_PATH)
        self.random_ps_notes_index = load_index(RANDOM_PS_NOTES_INDEX_PATH)
        self.ref_ps_index = load_index(REF_PS_INDEX_PATH)
        self.analyzed_index = load_index(ANALYZED_INDEX_PATH)
        self.content_index = load_index(CONTENT_INDEX_PATH)
        self.ref_ps_analysis = load_reference_ps_analysis(REF_PS_ANALYSIS_FILE)

    def search_content(self, index: VectorStoreIndex, query: str) -> str:
        """Search for content in the given index."""
        try:
            query_engine = index.as_query_engine()
            response = query_engine.query(query)
            return str(response)
        except Exception as e:
            logger.error(f"Failed to search content: {e}", exc_info=True)
            return ""

    @retry_on_error()
    def modify_personal_statement(self) -> str:
        """Modify the main personal statement based on tasks, comments, and generated content."""
        logger.info("Modifying personal statement")

        try:
            # Step 1: Generate initial draft
            initial_draft = self.generate_initial_draft()
            
            # Step 2: Verify initial draft
            if not self.verify_draft(initial_draft):
                logger.warning("Initial draft verification failed. Regenerating...")
                initial_draft = self.generate_initial_draft()
            
            # Step 3: Refine personal statement
            refined_ps = self.refine_personal_statement(initial_draft)
            
            # Step 4: Verify refined statement
            if not self.verify_refined_statement(refined_ps):
                logger.warning("Refined statement verification failed. Adjusting...")
                refined_ps = self.adjust_refined_statement(refined_ps)
            
            # Step 5: Final check and adjustments
            final_ps = self.final_check_and_adjust(refined_ps)
            
            return final_ps
        except Exception as e:
            logger.error(f"An error occurred while modifying the personal statement: {e}", exc_info=True)
            return self.main_ps  # Return the original statement if an error occurs

    # Add the other methods (generate_initial_draft, verify_draft, refine_personal_statement, etc.) here
    # Make sure to update their signatures to use self instead of passing individual parameters

    @retry_on_error()
    def verify_draft(self, draft: str) -> bool:
        """Verify if the initial draft meets the requirements."""
        try:
            prompt = f"""
            Original Personal Statement:
            {self.main_ps}

            Comments:
            {self.comments}

            Tasks:
            {self.tasks}

            Generated Draft:
            {draft}

            Please verify if the generated draft meets the following criteria:
            1. Addresses the comments and tasks
            2. Maintains the essence of the original personal statement
            3. Is between 750-800 words
            4. Is coherent and well-structured

            Respond with 'True' if all criteria are met, or 'False' if any are not met.
            """

            response = self.llm.complete(prompt)
            return response.text.strip().lower() == 'true'
        except Exception as e:
            logger.error(f"Failed to verify draft: {e}", exc_info=True)
            return False

    @retry_on_error()
    def verify_refined_statement(self, refined_ps: str) -> bool:
        """Verify if the refined personal statement meets all requirements and implements necessary changes."""
        try:
            prompt = f"""
            Original Personal Statement:
            {self.main_ps}

            Comments:
            {self.comments}

            Tasks:
            {self.tasks}

            Query Results:
            {self.search_content(self.content_index, "Provide relevant information for improving the personal statement.")}

            Refined Personal Statement:
            {refined_ps}

            Please verify if the refined personal statement meets the following criteria:
            1. Addresses all comments and tasks
            2. Incorporates relevant information from the query results
            3. Shows significant improvements over the original statement
            4. Maintains the essence and personal voice of the original statement while making necessary changes
            5. Is between 750-800 words
            6. Is coherent, well-structured, and flows logically
            7. Highlights key experiences and motivations for pediatrics
            8. Concludes strongly, tying together main themes

            Provide a detailed analysis of how well the refined statement meets each criterion, and respond with 'True' if all criteria are met, or 'False' if any are not met.
            """

            response = self.llm.complete(prompt)
            analysis = response.text.strip()
            logger.info(f"Verification analysis:\n{analysis}")
            return analysis.lower().endswith('true')
        except Exception as e:
            logger.error(f"Failed to verify refined statement: {e}", exc_info=True)
            return False

    @retry_on_error()
    def generate_initial_draft(self) -> str:
        supporting_info = self.search_content(self.supporting_docs_index, "Provide relevant information for improving the personal statement.")
        prioritized_info = self.search_content(self.prioritized_excerpts_index, "Provide relevant information for improving the personal statement.")
        random_notes_info = self.search_content(self.random_ps_notes_index, "Provide relevant information for improving the personal statement.")
        ref_ps_info = self.search_content(self.ref_ps_index, "Provide relevant information for improving the personal statement.")
        analyzed_info = self.search_content(self.analyzed_index, "Provide relevant information for improving the personal statement.")
        content_info = self.search_content(self.content_index, "Provide relevant information for improving the personal statement.")

        prompt = f"""
        Original Personal Statement:
        {self.main_ps}

        Comments:
        {self.comments}

        Tasks:
        {self.tasks}

        Generated Content:
        {json.dumps(self.content, indent=2)}

        Supporting Documents Information:
        {supporting_info}

        Prioritized Excerpts:
        {prioritized_info}

        Random Notes:
        {random_notes_info}

        Reference Personal Statements Information:
        {ref_ps_info}

        Analysis of Original Statement:
        {analyzed_info}

        Additional Content Information:
        {content_info}

        Reference PS Analysis:
        {json.dumps(self.ref_ps_analysis, indent=2)}

        Your task is to create a strong, cohesive personal statement for a pediatrics residency application. 
        Use the provided information as sources for content and improvements, but maintain the overall structure and flow of the original statement. 
        Ensure that the changes are primarily based on the original statement and align with the applicant's experiences.

        Follow these guidelines:
        1. Create a compelling opening hook that engages the reader.
        2. Incorporate relevant information from all provided sources, prioritizing the original statement and the applicant's actual experiences.
        3. Address the comments and fulfill the tasks.
        4. Maintain the personal voice and experiences of the original statement.
        5. Ensure the statement is cohesive, well-structured, and flows logically.
        6. Use the reference personal statements analysis as a guide for refinement.
        7. Consider the analysis of the original statement to address any weaknesses.
        8. Include specific anecdotes and experiences that demonstrate the applicant's passion for pediatrics.
        9. Highlight key skills, qualities, and experiences relevant to a pediatrics residency.
        10. Conclude with a strong statement that ties together the main themes and looks towards the future.
        11. Aim for a word count between 750-800 words.

        Please provide the complete, refined personal statement:
        """

        response = self.llm.complete(prompt)
        return response.text.strip()

    def refine_personal_statement(self, initial_draft: str) -> str:
        # Step 1: Adjust length
        adjusted_draft = self.adjust_personal_statement_length(initial_draft)
        
        # Step 2: Improve style
        styled_draft = self.improve_style(adjusted_draft)
        
        # Step 3: Ensure diversity in structure and vocabulary
        diverse_draft = self.diversify_structure_and_vocabulary(styled_draft)
        
        # Step 4: Grammar and language check
        checked_draft = self.grammar_check(diverse_draft)
        
        # Step 5: Final content and coherence check
        final_draft = self.final_content_check(checked_draft)
        
        return final_draft

    def final_content_check(self, ps: str) -> str:
        prompt = f"""
        Please perform a final check on this personal statement for a pediatrics residency application. 
        Ensure it meets the following criteria:

        1. Clearly communicates the applicant's passion for pediatrics
        2. Highlights relevant experiences and skills
        3. Demonstrates a clear understanding of the field
        4. Shows personal growth and reflection
        5. Is coherent and well-structured
        6. Ends with a strong conclusion

        If any improvements are needed, please make them while maintaining the current word count.

        Personal Statement:
        {ps}

        Provide the final, refined version of the personal statement:
        """
        
        response = self.llm.complete(prompt)
        return response.text.strip()

    @retry_on_error()
    def adjust_personal_statement_length(self, ps: str, target_min: int = 750, target_max: int = 800, max_attempts: int = 5) -> str:
        word_count = len(ps.split())
        
        if target_min <= word_count <= target_max:
            return ps
        
        for attempt in range(max_attempts):
            if word_count < target_min:
                ps = self.expand_personal_statement(ps, target_min, target_max)
            elif word_count > target_max:
                ps = self.trim_personal_statement(ps, target_min, target_max)
            
            word_count = len(ps.split())
            if target_min <= word_count <= target_max:
                logger.info(f"Successfully adjusted personal statement to {word_count} words.")
                return ps
            
            logger.warning(f"Adjustment attempt {attempt + 1} resulted in {word_count} words. Retrying...")
        
        logger.error("All adjustment attempts failed. Initiating fallback procedure.")
        return self.fallback_length_adjustment(ps, target_min, target_max)

    def expand_personal_statement(self, ps: str, target_min: int, target_max: int) -> str:
        expansion_prompt = f"""
        The personal statement is currently {len(ps.split())} words and needs to be expanded to between {target_min}-{target_max} words. 
        Please expand the content while maintaining coherence, relevance, and the applicant's voice. Follow these guidelines:

        1. Add specific details, examples, or elaborate on key experiences and motivations.
        2. Ensure each paragraph flows logically and connects to the overall narrative.
        3. Maintain the original tone and style of the applicant.
        4. Focus on depth rather than breadth - expand on existing points rather than introducing new ones.
        5. Verify that the expanded content aligns with the applicant's actual experiences and goals.

        Current personal statement:
        {ps}

        Provide an expanded version of the personal statement, adhering to the word count requirement:
        """
        expansion_response = self.llm.complete(expansion_prompt)
        return expansion_response.text.strip()

    def trim_personal_statement(self, ps: str, target_min: int, target_max: int) -> str:
        trim_prompt = f"""
        The personal statement is currently {len(ps.split())} words and needs to be trimmed to between {target_min}-{target_max} words. 
        Please reduce the content while maintaining coherence, relevance, and key information. Follow these guidelines:

        1. Remove redundant or less important information.
        2. Condense verbose phrases without losing meaning.
        3. Ensure the main themes and experiences are preserved.
        4. Maintain the overall flow and structure of the statement.

        Current personal statement:
        {ps}

        Provide a trimmed version of the personal statement, adhering to the word count requirement:
        """
        trim_response = self.llm.complete(trim_prompt)
        return trim_response.text.strip()

    def fallback_length_adjustment(self, ps: str, target_min: int, target_max: int) -> str:
        words = ps.split()
        if len(words) < target_min:
            words.extend([""] * (target_min - len(words)))
        elif len(words) > target_max:
            words = words[:target_max]
        return ' '.join(words)

    def fallback_expansion(self, modified_ps: str) -> str:
        logger.info("Initiating fallback expansion procedure.")
        words = modified_ps.split()
        current_count = len(words)
        target_count = 775

        if current_count < 750:
            # If too short, add generic sentences until reaching the minimum
            generic_sentences = [
                "This experience reinforced my commitment to pediatrics.",
                "I look forward to further developing my skills in this area.",
                "I am excited about the opportunities that lie ahead in my medical career.",
                "My passion for working with children continues to grow with each experience.",
                "I am eager to contribute to the field of pediatrics and make a positive impact."
            ]
            while len(words) < 750:
                words.append(generic_sentences[len(words) % len(generic_sentences)])
        elif current_count > 800:
            # If too long, truncate to 800 words
            words = words[:800]

        fallback_ps = ' '.join(words)
        logger.warning(f"Fallback procedure completed. Final word count: {len(fallback_ps.split())}")
        return fallback_ps

    @retry_on_error()
    def trim_personal_statement(self, ps: str, target_min: int, target_max: int) -> str:
        trim_prompt = f"""
        The personal statement is currently {len(ps.split())} words and needs to be trimmed to between {target_min}-{target_max} words. 
        Please reduce the content while maintaining coherence, relevance, and key information. Follow these guidelines:

        1. Remove redundant or less important information.
        2. Condense verbose phrases without losing meaning.
        3. Ensure the main themes and experiences are preserved.
        4. Maintain the overall flow and structure of the statement.

        Current personal statement:
        {ps}

        Original personal statement (for reference):
        {self.main_ps}

        Provide a trimmed version of the personal statement, adhering to the word count requirement:
        """
        trim_response = self.llm.complete(trim_prompt)
        trimmed_ps = trim_response.text.strip()
        
        if target_min <= len(trimmed_ps.split()) <= target_max:
            return trimmed_ps
        else:
            logger.warning(f"Trimming attempt failed to meet word count requirements. Manually trimming to {target_max} words.")
            return ' '.join(ps.split()[:target_max])

    @retry_on_error()
    def adjust_refined_statement(self, refined_ps: str) -> str:
        """Adjust the refined personal statement to meet all requirements and implement necessary changes."""
        prompt = f"""
        Original Personal Statement:
        {self.main_ps}

        Comments:
        {self.comments}

        Tasks:
        {self.tasks}

        Query Results:
        {self.search_content(self.content_index, "Provide relevant information for improving the personal statement.")}

        Refined Personal Statement:
        {refined_ps}

        The refined personal statement does not meet all requirements or implement necessary changes. Please adjust it to:
        1. Address all comments and tasks thoroughly
        2. Incorporate relevant information from the query results
        3. Show significant improvements over the original statement
        4. Maintain the essence and personal voice of the original statement while making necessary changes
        5. Be between 750-800 words
        6. Be coherent, well-structured, and flow logically
        7. Highlight key experiences and motivations for pediatrics
        8. Conclude strongly, tying together main themes

        Please provide the adjusted personal statement, ensuring that it demonstrates clear improvements and addresses all required changes:
        """

        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to adjust refined statement: {e}", exc_info=True)
            return refined_ps

    @retry_on_error()
    def final_check_and_adjust(self, ps: str) -> str:
        """Perform a final check and make any necessary adjustments to ensure all requirements are met."""
        prompt = f"""
        Original Personal Statement:
        {self.main_ps}

        Comments:
        {self.comments}

        Tasks:
        {self.tasks}

        Query Results:
        {self.search_content(self.content_index, "Provide relevant information for improving the personal statement.")}

        Current Personal Statement:
        {ps}

        Please perform a final check on the personal statement and make any necessary adjustments to ensure:
        1. All comments and tasks are fully addressed
        2. Relevant information from query results is incorporated
        3. The statement shows significant improvements over the original
        4. The essence of the original statement is maintained while implementing necessary changes
        5. The word count is between 750-800 words
        6. The statement is coherent, well-structured, and flows logically
        7. Key experiences and motivations for pediatrics are highlighted
        8. The conclusion ties together the main themes effectively

        If adjustments are needed, please provide the final version of the personal statement. 
        If no adjustments are needed, respond with "No adjustments needed."

        Additionally, provide a brief summary of the improvements made and how they address the original requirements.
        """

        try:
            response = self.llm.complete(prompt)
            result = response.text.strip()
            if result != "No adjustments needed.":
                logger.info(f"Final adjustments made:\n{result}")
                return result.split("\n\nImprovements summary:")[0]
            else:
                logger.info("No final adjustments needed.")
                return ps
        except Exception as e:
            logger.error(f"Failed to perform final check and adjust: {e}", exc_info=True)
            return ps

    def modify_personal_statement(self) -> List[str]:
        """Modify the main personal statement and generate three different drafts."""
        logger.info("Modifying personal statement and generating three drafts")

        try:
            drafts = []
            draft_types = ["regular", "experimental1", "experimental2"]
            max_attempts = 3

            for draft_type in draft_types:
                draft = self.generate_draft(draft_type)
                refined_draft = self.refine_personal_statement(draft)
                
                for attempt in range(max_attempts):
                    if self.verify_draft_quality(refined_draft):
                        drafts.append(refined_draft)
                        logger.info(f"Successfully generated and verified {draft_type} draft")
                        break
                    elif attempt < max_attempts - 1:
                        logger.warning(f"{draft_type.capitalize()} draft failed quality check. Attempting improvement.")
                        refined_draft = self.improve_draft_quality(refined_draft)
                    else:
                        logger.error(f"Failed to generate acceptable {draft_type} draft after {max_attempts} attempts.")
                        raise ValueError(f"Failed to generate acceptable {draft_type} draft")

            if len(drafts) != 3:
                raise ValueError(f"Expected 3 drafts, but generated {len(drafts)}")

            logger.info("Successfully generated three quality-verified drafts")
            return drafts

        except Exception as e:
            logger.error(f"An error occurred while modifying the personal statement: {e}", exc_info=True)
            raise

    def generate_draft(self, draft_type: str) -> str:
        if draft_type == "regular":
            return self.generate_regular_draft()
        else:
            return self.generate_experimental_draft(int(draft_type[-1]))

    def generate_regular_draft(self) -> str:
        """Generate the regular draft of the personal statement."""
        max_attempts = 3
        for attempt in range(max_attempts):
            initial_draft = self.generate_initial_draft()
            refined_ps = self.refine_personal_statement(initial_draft)
            refined_ps = self.improve_style(refined_ps)
            refined_ps = self.diversify_structure_and_vocabulary(refined_ps)
            refined_ps = self.grammar_check(refined_ps)
            final_ps = self.final_check_and_adjust(refined_ps)
            final_ps = ensure_original_content(self.main_ps, final_ps)
            
            if self.verify_draft_quality(final_ps):
                return final_ps
            
            logger.warning(f"Draft {attempt + 1} did not meet quality standards. Retrying...")
        
        raise ValueError("Failed to generate a satisfactory draft after multiple attempts.")

    def generate_experimental_draft(self, draft_number: int) -> str:
        """Generate an experimental draft of the personal statement."""
        max_attempts = 3
        for attempt in range(max_attempts):
            prompt = f"""
            Generate an experimental version (Draft {draft_number}) of the personal statement that is more dissimilar 
            from the regular output. This version should still be based on the user's content and experiences, 
            but with a different style, structure, and approach. Use the reference personal statements and analysis 
            for guidance, but ensure the content remains true to the user's experiences.

            Original Personal Statement:
            {self.main_ps}

            Reference PS Analysis:
            {json.dumps(self.ref_ps_analysis, indent=2)}

            Create a unique and compelling personal statement that:
            1. Has a different structure and flow compared to the original
            2. Uses a distinct writing style
            3. Highlights different aspects of the applicant's experiences
            4. Maintains the core content and motivations
            5. Is between 750-800 words
            6. Avoids clichés, cheesy language, and overused phrases
            7. Presents a genuine and authentic voice

            Provide the experimental personal statement:
            """
            
            response = self.llm.complete(prompt)
            experimental_draft = response.text.strip()
            refined_draft = self.refine_personal_statement(experimental_draft)
            
            if self.verify_draft_quality(refined_draft):
                return refined_draft
            
            logger.warning(f"Experimental draft {draft_number}, attempt {attempt + 1} did not meet quality standards. Retrying...")
        
        raise ValueError(f"Failed to generate a satisfactory experimental draft {draft_number} after multiple attempts.")

    def save_modified_ps(self, modified_ps_list: List[str]):
        """Save the modified personal statements to files."""
        logger.info(f"Saving modified personal statements to {OUTPUT_FOLDER}")
        try:
            # Create a unique folder name with timestamp and counter
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            counter = 1
            while True:
                folder_name = f"{timestamp}_{counter:03d}"
                output_dir = os.path.join(OUTPUT_FOLDER, folder_name)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    break
                counter += 1
            
            for i, modified_ps in enumerate(modified_ps_list):
                draft_name = f"draft_{i+1}"
                
                # Save the modified personal statement as .txt
                txt_path = os.path.join(output_dir, f"{draft_name}.txt")
                with open(txt_path, 'w') as f:
                    f.write(modified_ps)
                
                # Save the modified personal statement as .json
                json_path = os.path.join(output_dir, f"{draft_name}.json")
                with open(json_path, 'w') as f:
                    json.dump({"personal_statement": modified_ps}, f, indent=2)
            
            logger.info(f"Modified personal statements saved successfully to {output_dir}")
        except Exception as e:
            logger.error(f"Failed to save modified personal statements: {e}", exc_info=True)

    @retry_on_error()
    def intensive_refinement(self, ps: str) -> str:
        """Perform intensive refinement on the personal statement."""
        logger.info("Initiating intensive refinement process")
        
        # Step 1: Style analysis and improvement
        ps = self.improve_style(ps)
        
        # Step 2: Ensure diversity in sentence structure and vocabulary
        ps = self.diversify_structure_and_vocabulary(ps)
        
        # Step 3: Grammar and language check
        ps = self.grammar_check(ps)
        
        # Step 4: Final LLM-based refinement
        prompt = f"""
        Please perform a final, intensive refinement on this personal statement for a pediatrics residency application. 
        Focus on making it more compelling, unique, and tailored to pediatrics while maintaining the applicant's voice and experiences. 
        Ensure it stands out from typical personal statements while remaining authentic and true to the applicant's journey.

        Original Personal Statement:
        {self.main_ps}

        Current Personal Statement:
        {ps}

        Provide the final, refined version of the personal statement:
        """
        
        response = self.llm.complete(prompt)
        refined_ps = response.text.strip()
        
        return refined_ps

    @retry_on_error()
    def improve_style(self, ps: str) -> str:
        """Improve the style of the personal statement."""
        prompt = f"""
        Analyze and improve the style of this personal statement for a pediatrics residency application. 
        Focus on enhancing clarity, coherence, and impact while maintaining the applicant's voice. 
        Ensure a good balance between personal anecdotes and professional aspirations.

        Personal Statement:
        {ps}

        Provide the improved version:
        """
        
        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error in improve_style: {e}")
            return ps  # Return original text if improvement fails

    def diversify_structure_and_vocabulary(self, ps: str) -> str:
        """Ensure diversity in sentence structure and vocabulary."""
        sentences = sent_tokenize(ps)
        
        prompt = f"""
        Analyze the following sentences from a personal statement and suggest improvements to diversify sentence structure and vocabulary. 
        Maintain the original meaning and tone, but make the writing more engaging and varied.

        Sentences:
        {json.dumps(sentences)}

        Provide the improved sentences, one per line:
        """
        
        response = self.llm.complete(prompt)
        improved_sentences = response.text.strip().split('\n')
        
        return ' '.join(improved_sentences)

    @retry_on_error(max_retries=3, delay=5)
    def grammar_check(self, ps: str) -> str:
        """Perform a grammar and language check on the personal statement."""
        prompt = f"""
        Please review the following personal statement for any grammatical issues or language improvements. 
        Correct any errors and enhance the language while maintaining the original meaning and tone:

        Personal Statement:
        {ps}

        Provide the corrected and improved version:
        """
        
        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error in grammar check: {e}")
            return ps  # Return original text if grammar check fails

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate the cosine similarity between two texts."""
        stop_words = set(stopwords.words('english'))
        
        def preprocess(text):
            tokens = word_tokenize(text.lower())
            return ' '.join([word for word in tokens if word.isalnum() and word not in stop_words])
        
        preprocessed_text1 = preprocess(text1)
        preprocessed_text2 = preprocess(text2)
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([preprocessed_text1, preprocessed_text2])
        
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def main():
    logger.info("Starting personal statement modification process")
    try:
        # Set your Anthropic API key
        os.environ["ANTHROPIC_API_KEY"] = ""
        logger.info("Anthropic API key set")

        agent = PersonalStatementAgent()
        agent.load_data()

        # Modify personal statement and generate three drafts
        modified_ps_list = agent.modify_personal_statement()

        # Save modified personal statements
        agent.save_modified_ps(modified_ps_list)

        logger.info("Personal statement modification process completed successfully")
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}", exc_info=True)

if __name__ == "__main__":
    main()
def search_content(index: VectorStoreIndex, query: str) -> str:
    """Search for content in the given index."""
    try:
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        logger.error(f"Failed to search content: {e}", exc_info=True)
        return ""
def verify_draft(draft: str, main_ps: str, comments: str, tasks: str) -> bool:
    """Verify if the initial draft meets the requirements."""
    llm = Anthropic(model=LLM_MODEL)
    
    prompt = f"""
    Original Personal Statement:
    {main_ps}

    Comments:
    {comments}

    Tasks:
    {tasks}

    Generated Draft:
    {draft}

    Please verify if the generated draft meets the following criteria:
    1. Addresses the comments and tasks
    2. Maintains the essence of the original personal statement
    3. Is between 750-800 words
    4. Is coherent and well-structured

    Respond with 'True' if all criteria are met, or 'False' if any are not met.
    """

    try:
        response = llm.complete(prompt)
        return response.text.strip().lower() == 'true'
    except Exception as e:
        logger.error(f"Failed to verify draft: {e}", exc_info=True)
        return False

def verify_refined_statement(llm: Anthropic, refined_ps: str, main_ps: str, comments: str, tasks: str) -> bool:
    """Verify if the refined personal statement meets all requirements."""
    try:
        prompt = f""" 
        Original Personal Statement:
        {main_ps}

        Comments:
        {comments}

        Tasks:
        {tasks}

        Refined Personal Statement:
        {refined_ps}

        Please verify if the refined personal statement meets the following criteria:
        1. Addresses all comments and tasks
        2. Maintains the essence and personal voice of the original statement
        3. Is between 750-800 words
        4. Is coherent, well-structured, and flows logically
        5. Highlights key experiences and motivations for pediatrics
        6. Concludes strongly, tying together main themes

        Respond with 'True' if all criteria are met, or 'False' if any are not met.
        """

        response = llm.complete(prompt)
        return response.text.strip().lower() == 'true'
    except Exception as e:
        logger.error(f"Failed to verify refined statement: {e}", exc_info=True)
        return False

def adjust_refined_statement(llm: Anthropic, refined_ps: str, main_ps: str, comments: str, tasks: str) -> str:
    """Adjust the refined personal statement to meet all requirements."""
    prompt = f"""
    Original Personal Statement:
    {main_ps}

    Comments:
    {comments}

    Tasks:
    {tasks}

    Refined Personal Statement:
    {refined_ps}

    The refined personal statement does not meet all requirements. Please adjust it to:
    1. Address all comments and tasks
    2. Maintain the essence and personal voice of the original statement
    3. Be between 750-800 words
    4. Be coherent, well-structured, and flow logically
    5. Highlight key experiences and motivations for pediatrics
    6. Conclude strongly, tying together main themes

    Please provide the adjusted personal statement:
    """

    try:
        response = llm.complete(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Failed to adjust refined statement: {e}", exc_info=True)
        return refined_ps

def final_check_and_adjust(llm: Anthropic, ps: str, main_ps: str, comments: str, tasks: str) -> str:
    """Perform a final check and make any necessary adjustments."""
    prompt = f"""
    Original Personal Statement:
    {main_ps}

    Comments:
    {comments}

    Tasks:
    {tasks}

    Current Personal Statement:
    {ps}

    Please perform a final check on the personal statement and make any necessary adjustments to ensure:
    1. All comments and tasks are fully addressed
    2. The essence of the original statement is maintained
    3. The word count is between 750-800 words
    4. The statement is coherent, well-structured, and flows logically
    5. Key experiences and motivations for pediatrics are highlighted
    6. The conclusion ties together the main themes effectively

    If adjustments are needed, please provide the final version of the personal statement. 
    If no adjustments are needed, respond with "No adjustments needed."
    """

    try:
        response = llm.complete(prompt)
        result = response.text.strip()
        return ps if result == "No adjustments needed." else result
    except Exception as e:
        logger.error(f"Failed to perform final check and adjust: {e}", exc_info=True)
        return ps

    def improve_draft_quality(self, draft: str) -> str:
        """Improve the quality of the draft based on the verification feedback."""
        prompt = f"""
        The following personal statement draft for a pediatrics residency application needs improvement. Please refine it to meet these criteria:

        1. Maintain a word count between 770-780 words
        2. Remove any clichés, cheesy language, or corny expressions
        3. Enhance the genuine and authentic voice
        4. Improve coherence and structure
        5. Strengthen the communication of specific motivations and experiences relevant to pediatrics
        6. Ensure it's tailored explicitly for a pediatrics residency
        7. Add more detailed, specific examples
        8. Incorporate unique insights or perspectives related to pediatrics
        9. Refine grammar, sentence structure, and vocabulary
        10. Enhance the overall impact and memorability

        Current draft:
        {draft}

        Please provide an improved version of the personal statement:
        """

        try:
            response = self.llm.complete(prompt)
            improved_draft = response.text.strip()
            logger.info("Draft quality improvement attempt completed.")
            return improved_draft
        except Exception as e:
            logger.error(f"Failed to improve draft quality: {e}", exc_info=True)
            return draft

    @retry_on_error()
    def verify_draft_quality(self, draft: str) -> bool:
        """Verify if the draft meets all quality requirements."""
        word_count = len(draft.split())
        if word_count < 770 or word_count > 780:
            logger.warning(f"Draft word count ({word_count}) is outside the required range of 770-780 words.")
            return False

        prompt = f"""
        Analyze this personal statement draft for a pediatrics residency application with extreme scrutiny. Determine if it meets ALL of the following criteria:

        1. Word count: {word_count} (must be between 770-780 words)
        2. Content: Entirely free of clichés, cheesy language, or corny expressions
        3. Voice: Presents a genuine, authentic, and unique voice
        4. Structure: Coherent, well-structured, with clear flow between paragraphs
        5. Communication: Effectively conveys the applicant's specific motivations and relevant experiences for pediatrics
        6. Relevance: Tailored explicitly for a pediatrics residency application
        7. Depth: Provides detailed, specific examples rather than general statements
        8. Originality: Offers unique insights or perspectives related to pediatrics
        9. Grammar and style: Impeccable grammar, varied sentence structure, and appropriate vocabulary
        10. Impact: Leaves a strong, memorable impression about the applicant's fit for pediatrics

        Draft:
        {draft}

        Respond with 'True' ONLY if ALL criteria are fully met without exception. Otherwise, respond with 'False' and provide a concise, bullet-point list of unmet criteria and brief explanations.
        """

        try:
            response = self.llm.complete(prompt)
            result = response.text.strip().lower()
            if not result.startswith('true'):
                logger.warning(f"Draft quality check failed:\n{result}")
                return False
            logger.info("Draft passed quality check successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to verify draft quality: {e}", exc_info=True)
            return False

class PersonalStatementAgent:
    def __init__(self):
        self.llm = Anthropic(model=LLM_MODEL, max_tokens=4000)
        self.main_ps = ""
        self.comments = ""
        self.tasks = ""
        self.content = []
        self.supporting_docs_index = None
        self.prioritized_excerpts_index = None
        self.random_ps_notes_index = None
        self.ref_ps_index = None
        self.analyzed_index = None
        self.content_index = None
        self.ref_ps_analysis = {}

    # ... (other methods)

    @retry_on_error()
    def verify_draft_quality(self, draft: str) -> bool:
        """Verify if the draft meets all quality requirements."""
        word_count = len(draft.split())
        if word_count < 770 or word_count > 780:
            logger.warning(f"Draft word count ({word_count}) is outside the required range of 770-780 words.")
            return False

        prompt = f"""
        Analyze this personal statement draft for a pediatrics residency application with extreme scrutiny. Determine if it meets ALL of the following criteria:

        1. Word count: {word_count} (must be between 770-780 words)
        2. Content: Entirely free of clichés, cheesy language, or corny expressions
        3. Voice: Presents a genuine, authentic, and unique voice
        4. Structure: Coherent, well-structured, with clear flow between paragraphs
        5. Communication: Effectively conveys the applicant's specific motivations and relevant experiences for pediatrics
        6. Relevance: Tailored explicitly for a pediatrics residency application
        7. Depth: Provides detailed, specific examples rather than general statements
        8. Originality: Offers unique insights or perspectives related to pediatrics
        9. Grammar and style: Impeccable grammar, varied sentence structure, and appropriate vocabulary
        10. Impact: Leaves a strong, memorable impression about the applicant's fit for pediatrics

        Draft:
        {draft}

        Respond with 'True' ONLY if ALL criteria are fully met without exception. Otherwise, respond with 'False' and provide a concise, bullet-point list of unmet criteria and brief explanations.
        """

        try:
            response = self.llm.complete(prompt)
            result = response.text.strip().lower()
            if not result.startswith('true'):
                logger.warning(f"Draft quality check failed:\n{result}")
                return False
            logger.info("Draft passed quality check successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to verify draft quality: {e}", exc_info=True)
            return False

def search_content(index: VectorStoreIndex, query: str) -> str:
    """Search for content in the given index."""
    try:
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        logger.error(f"Failed to search content: {e}", exc_info=True)
        return ""
