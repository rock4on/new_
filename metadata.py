
import re

import os

import json

from openai import OpenAI

from typing import Dict, List, Optional

from langdetect import detect

from dotenv import load_dotenv

import sys

from pydantic import BaseModel, Field, validator

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

 

from config import OpenAiCredentials

 

DEFAULT_MODEL = "vertex_ai.gemini-2.0-flash-lite"

 

client = OpenAI(api_key=OpenAiCredentials.api_key, base_url=OpenAiCredentials.base_url)

 

class ESGMetadata(BaseModel):
    country: str = Field(..., description="Country where the regulation applies")
    jurisdiction: str = Field(..., description="Jurisdiction or region of the regulation")
    issuing_body: str = Field(..., description="Organization or body that issued the regulation")
    tag: str = Field(..., description="Classification tag for the regulation")
    regulation_name: str = Field(..., description="Official name of the regulation")
    publication_date: str = Field(..., description="Date when the regulation was published")
    regulation_status: str = Field(..., description="Current status of the regulation")
    summary: str = Field(..., description="Brief summary of the regulation")
    applicability: str = Field(..., description="Who or what the regulation applies to")
    scoping_threshold: str = Field(..., description="Threshold criteria for regulation scope")
    filing_mechanism: str = Field(..., description="How to file or comply with the regulation")
    reporting_frequency: str = Field(..., description="How often reporting is required")
    assurance_requirement: str = Field(..., description="Assurance or verification requirements")
    full_text_link: str = Field(..., description="Link to the full text of the regulation")
    translated_flag: str = Field(default="no", description="Whether the text is translated")
    source_url: str = Field(..., description="URL of the source document")
    last_scraped: str = Field(..., description="Date when the document was last scraped")
    change_detected: str = Field(..., description="Whether changes were detected")

    @validator('translated_flag')
    def validate_translated_flag(cls, v):
        if v is None:
            return 'no'
        if str(v).lower() not in ['yes', 'no']:
            return 'no'
        return str(v).lower()

    @validator('*', pre=True)
    def ensure_string(cls, v):
        if v is None:
            return "Unknown"
        return str(v)


class LightMetadata(BaseModel):
    metadata_type: str = Field(default="light", description="Type of metadata extraction")
    esg_relevant: bool = Field(default=True, description="Whether the document is ESG relevant")
    match_score: int = Field(..., description="Score based on keyword matches")
    source_url: str = Field(..., description="URL of the source document")
    last_scraped: str = Field(..., description="Date when the document was last scraped")


class FullMetadata(ESGMetadata):
    metadata_type: str = Field(default="full", description="Type of metadata extraction")
    match_score: int = Field(..., description="Score based on keyword matches")


class PartialMetadata(BaseModel):
    metadata_type: str = Field(default="partial", description="Type of metadata extraction")
    source_url: str = Field(..., description="URL of the source document")
    last_scraped: str = Field(..., description="Date when the document was last scraped")
    match_score: int = Field(..., description="Score based on keyword matches")


REQUIRED_METADATA_FIELDS = [

    "country", "jurisdiction", "issuing_body", "tag", "regulation_name", "publication_date",

    "regulation_status", "summary", "applicability", "scoping_threshold", "filing_mechanism",

    "reporting_frequency", "assurance_requirement", "full_text_link", "translated_flag",

    "source_url", "last_scraped", "change_detected"

]

 

def translate_keywords_with_llm(keywords: List[str], target_lang: str) -> List[str]:

    if target_lang.lower() in ['en', 'english']:

        return keywords

 

    system_prompt = "You are a translator that translates ESG-related keywords into the target language."

    user_prompt = f"""

Translate the following ESG keywords into {target_lang}.

Return ONLY a JSON array. Do NOT include markdown or explanations.

 

Keywords: {keywords}

"""

 

    try:

        response = client.chat.completions.create(

            model="vertex_ai.gemini-2.0-flash-lite",

            messages=[

                {"role": "system", "content": system_prompt},

                {"role": "user", "content": user_prompt}

            ],

            temperature=0

        )

        content = response.choices[0].message.content

        print(f"[TRANSLATION RAW OUTPUT] {content}")

 

        # 🛠 STRIP markdown if present (e.g., ```json ... ```)

        match = re.search(r'\[.*?\]', content, re.DOTALL)

        if match:

            content = match.group(0)

 

        translated = json.loads(content)

        if isinstance(translated, list):

            return [kw.lower() for kw in translated]

 

    except Exception as e:

        print(f"[Translation Error] {e}")

 

    return keywords  # fallback

 

def safe_json_parse(content: str) -> Dict[str, str]:

    try:

        return json.loads(content)

    except json.JSONDecodeError:

        match = re.search(r'\{.*\}', content, re.DOTALL)

        if match:

            try:

                return json.loads(match.group(0))

            except:

                pass

    return {}

 

def extract_snippets_around_keywords(text: str, keywords: List[str], window: int = 4000) -> List[str]:

    """Extract windows of context around ESG keyword matches."""

    snippets = []

    for kw in keywords:

        for match in re.finditer(re.escape(kw), text, flags=re.IGNORECASE):

            start = match.start()

            snippet = text[max(0, start - window): start + window]

            snippets.append(snippet)

    return snippets

 

def is_metadata_complete(meta: ESGMetadata) -> bool:

    # Check if metadata has meaningful content (not all "Unknown")
    unknown_fields = sum(1 for field in ['jurisdiction', 'issuing_body', 'tag', 'regulation_name', 
                                       'publication_date', 'regulation_status', 'summary', 
                                       'applicability', 'scoping_threshold', 'filing_mechanism',
                                       'reporting_frequency', 'assurance_requirement', 'full_text_link']
                        if getattr(meta, field) == "Unknown")
    
    # Consider complete if less than half the fields are "Unknown"
    return unknown_fields < 7

 

def run_llm(text: str, url: str, country: str, last_scraped: str) -> ESGMetadata:

    system_msg = (

        "You are an assistant that extracts ESG regulation metadata from documents.\n"

        "You must return a valid JSON object that matches this exact schema:\n"

        f"{ESGMetadata.model_json_schema()}\n"

        "Use 'translated_flag': 'no' unless the text is clearly translated. Return only the JSON."

    )

 

    user_msg = f"""

Document Source URL: {url}

Country: {country}

Last Scraped: {last_scraped}

 

Document Text:

{text}

"""

 

    try:

        response = client.chat.completions.create(

            model="vertex_ai.gemini-2.0-flash-lite",

            messages=[

                {"role": "system", "content": system_msg},

                {"role": "user", "content": user_msg}

            ],

            temperature=0,

        )

        content = response.choices[0].message.content

        

        # Parse JSON and validate with Pydantic

        parsed_data = safe_json_parse(content)

        if parsed_data:

            # Add required fields that are provided as parameters

            parsed_data['source_url'] = url

            parsed_data['last_scraped'] = last_scraped

            if 'country' not in parsed_data:

                parsed_data['country'] = country

            

            try:

                return ESGMetadata(**parsed_data)

            except Exception as validation_error:

                print(f"[VALIDATION ERROR] {validation_error}")

                # Return default metadata if validation fails

                return ESGMetadata(

                    country=country,

                    jurisdiction="Unknown",

                    issuing_body="Unknown",

                    tag="Unknown",

                    regulation_name="Unknown",

                    publication_date="Unknown",

                    regulation_status="Unknown",

                    summary="Unable to extract sufficient information",

                    applicability="Unknown",

                    scoping_threshold="Unknown",

                    filing_mechanism="Unknown",

                    reporting_frequency="Unknown",

                    assurance_requirement="Unknown",

                    full_text_link="Unknown",

                    source_url=url,

                    last_scraped=last_scraped,

                    change_detected="Unknown"

                )

        # Return default metadata if parsing fails

        return ESGMetadata(

            country=country,

            jurisdiction="Unknown",

            issuing_body="Unknown",

            tag="Unknown",

            regulation_name="Unknown",

            publication_date="Unknown",

            regulation_status="Unknown",

            summary="Unable to extract sufficient information",

            applicability="Unknown",

            scoping_threshold="Unknown",

            filing_mechanism="Unknown",

            reporting_frequency="Unknown",

            assurance_requirement="Unknown",

            full_text_link="Unknown",

            source_url=url,

            last_scraped=last_scraped,

            change_detected="Unknown"

        )

    except Exception as e:

        print(f"[LLM ERROR] {e}")

        # Return default metadata if LLM call fails

        return ESGMetadata(

            country=country,

            jurisdiction="Unknown",

            issuing_body="Unknown",

            tag="Unknown",

            regulation_name="Unknown",

            publication_date="Unknown",

            regulation_status="Unknown",

            summary="Unable to extract sufficient information",

            applicability="Unknown",

            scoping_threshold="Unknown",

            filing_mechanism="Unknown",

            reporting_frequency="Unknown",

            assurance_requirement="Unknown",

            full_text_link="Unknown",

            source_url=url,

            last_scraped=last_scraped,

            change_detected="Unknown"

        )

 

def extract_metadata(

    text: str,

    url: str,

    country: str,

    esg_keywords: List[str],

    last_scraped: str,

    min_match_score: int = 1

) -> Dict:

    """Hybrid metadata extractor with fallback logic for low-relevance documents."""

    #count key word hits

 

    lang = detect(text)

    translated_keywords = translate_keywords_with_llm(esg_keywords, lang)

    print(f"[ESG Match] Language: {lang}, Keywords used: {translated_keywords}")

 

    match_score = sum(len(re.findall(rf'\b{re.escape(k)}\b', text, flags=re.IGNORECASE)) for k in translated_keywords)

    print(f"[LANG DETECTED] {lang} — Match score: {match_score}")

 

    #return full metadata even for low match scores

    if match_score < min_match_score:

        # Try to extract whatever metadata we can

        metadata = run_llm(text, url, country, last_scraped)

        if metadata:

            full_metadata = FullMetadata(

                **metadata.model_dump(),

                match_score=match_score

            )

            return full_metadata.model_dump()

        else:

            # Create minimal FullMetadata with default values

            full_metadata = FullMetadata(

                country=country,

                jurisdiction="Unknown",

                issuing_body="Unknown",

                tag="Unknown",

                regulation_name="Unknown",

                publication_date="Unknown",

                regulation_status="Unknown",

                summary="Unable to extract sufficient information",

                applicability="Unknown",

                scoping_threshold="Unknown",

                filing_mechanism="Unknown",

                reporting_frequency="Unknown",

                assurance_requirement="Unknown",

                full_text_link="Unknown",

                source_url=url,

                last_scraped=last_scraped,

                change_detected="Unknown",

                match_score=match_score

            )

            return full_metadata.model_dump()

 

    # sample snippets around keywords

    snippets = extract_snippets_around_keywords(text, translated_keywords)

    for snippet in snippets:

        metadata = run_llm(snippet, url, country, last_scraped)

        if is_metadata_complete(metadata):

            full_metadata = FullMetadata(

                **metadata.model_dump(),

                match_score=match_score

            )

            return full_metadata.model_dump()

 

    # run full llm

    metadata = run_llm(text, url, country, last_scraped)

    if is_metadata_complete(metadata):

        full_metadata = FullMetadata(

            **metadata.model_dump(),

            match_score=match_score

        )

        return full_metadata.model_dump()

    else:

        # Create FullMetadata with default values for missing fields

        full_metadata = FullMetadata(

            country=country,

            jurisdiction="Unknown",

            issuing_body="Unknown",

            tag="Unknown",

            regulation_name="Unknown",

            publication_date="Unknown",

            regulation_status="Unknown",

            summary="Unable to extract sufficient information",

            applicability="Unknown",

            scoping_threshold="Unknown",

            filing_mechanism="Unknown",

            reporting_frequency="Unknown",

            assurance_requirement="Unknown",

            full_text_link="Unknown",

            source_url=url,

            last_scraped=last_scraped,

            change_detected="Unknown",

            match_score=match_score

        )

        return full_metadata.model_dump()

 

