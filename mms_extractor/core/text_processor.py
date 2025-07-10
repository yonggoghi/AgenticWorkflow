"""
Text processing module for cleaning and normalizing Korean text.
"""

from typing import List, Dict, Any, Set, Optional
import re
import unicodedata
from ..config.settings import TEXT_PROCESSING_CONFIG

class TextProcessor:
    """
    A class for processing and cleaning Korean text.
    
    This class handles text normalization, cleaning, and preprocessing
    specifically optimized for Korean text processing.
    """
    
    def __init__(
        self,
        tags_to_exclude: Set[str] = set(TEXT_PROCESSING_CONFIG["tags_to_exclude"]),
        space_tolerance: int = TEXT_PROCESSING_CONFIG["space_tolerance"]
    ):
        """
        Initialize the TextProcessor.
        
        Args:
            tags_to_exclude: Set of HTML/XML tags to remove
            space_tolerance: Maximum number of consecutive spaces to normalize
        """
        self.tags_to_exclude = tags_to_exclude
        self.space_tolerance = space_tolerance
        
        # Compile regex patterns
        self.tag_pattern = re.compile(
            r'<[^>]+>',
            re.IGNORECASE | re.MULTILINE
        )
        self.space_pattern = re.compile(
            r'\s{' + str(space_tolerance + 1) + r',}',
            re.MULTILINE
        )
        self.url_pattern = re.compile(
            r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',
            re.IGNORECASE
        )
        self.email_pattern = re.compile(
            r'[\w\.-]+@[\w\.-]+\.\w+',
            re.IGNORECASE
        )
        self.phone_pattern = re.compile(
            r'(?:\+82|0)\s*[-]?\s*(?:\d{1,2}|\(\d{1,2}\))\s*[-]?\s*\d{3,4}\s*[-]?\s*\d{4}'
        )
        
    def clean_text(
        self,
        text: str,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_phones: bool = True,
        normalize_spaces: bool = True,
        normalize_unicode: bool = True
    ) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            remove_phones: Whether to remove phone numbers
            normalize_spaces: Whether to normalize spaces
            normalize_unicode: Whether to normalize unicode characters
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Convert to string if not already
        text = str(text)
        
        # Remove HTML/XML tags
        text = self._remove_tags(text)
        
        # Remove URLs
        if remove_urls:
            text = self.url_pattern.sub(' ', text)
        
        # Remove email addresses
        if remove_emails:
            text = self.email_pattern.sub(' ', text)
        
        # Remove phone numbers
        if remove_phones:
            text = self.phone_pattern.sub(' ', text)
        
        # Normalize spaces
        if normalize_spaces:
            text = self.space_pattern.sub(' ', text)
            text = text.strip()
        
        # Normalize unicode
        if normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        return text
    
    def _remove_tags(self, text: str) -> str:
        """Remove HTML/XML tags from text."""
        def replace_tag(match):
            tag = match.group(0).lower()
            # Check if tag should be excluded
            for exclude_tag in self.tags_to_exclude:
                if tag.startswith(f'<{exclude_tag}'):
                    return ' '
            return ' '
        
        return self.tag_pattern.sub(replace_tag, text)
    
    def extract_sentences(
        self,
        text: str,
        min_length: int = 10,
        max_length: int = 500
    ) -> List[str]:
        """
        Extract sentences from text.
        
        Args:
            text: Input text
            min_length: Minimum sentence length
            max_length: Maximum sentence length
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Clean text first
        text = self.clean_text(text)
        
        # Split into sentences
        # Handle various sentence endings
        sentence_endings = r'[.!?…。！？]'
        sentences = re.split(
            f'({sentence_endings}\\s*)',
            text
        )
        
        # Combine sentences with their endings
        result = []
        current = ""
        
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                current = sentences[i] + sentences[i + 1]
            else:
                current = sentences[i]
            
            current = current.strip()
            
            if min_length <= len(current) <= max_length:
                result.append(current)
        
        # Handle last sentence if it exists
        if len(sentences) % 2 == 1:
            last = sentences[-1].strip()
            if min_length <= len(last) <= max_length:
                result.append(last)
        
        return result
    
    def extract_phrases(
        self,
        text: str,
        min_length: int = 3,
        max_length: int = 50,
        delimiters: Optional[str] = None
    ) -> List[str]:
        """
        Extract phrases from text.
        
        Args:
            text: Input text
            min_length: Minimum phrase length
            max_length: Maximum phrase length
            delimiters: Custom delimiters for phrase splitting
            
        Returns:
            List of phrases
        """
        if not text:
            return []
        
        # Clean text first
        text = self.clean_text(text)
        
        # Default delimiters for Korean text
        if delimiters is None:
            delimiters = r'[,，、;；:：]'
        
        # Split into phrases
        phrases = re.split(f'[{delimiters}]', text)
        
        # Filter and clean phrases
        result = []
        for phrase in phrases:
            phrase = phrase.strip()
            if min_length <= len(phrase) <= max_length:
                result.append(phrase)
        
        return result
    
    def normalize_korean_text(
        self,
        text: str,
        remove_diacritics: bool = True,
        normalize_spaces: bool = True
    ) -> str:
        """
        Normalize Korean text.
        
        Args:
            text: Input text
            remove_diacritics: Whether to remove diacritics
            normalize_spaces: Whether to normalize spaces
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove diacritics if requested
        if remove_diacritics:
            text = ''.join(
                c for c in text
                if not unicodedata.combining(c)
            )
        
        # Normalize spaces
        if normalize_spaces:
            text = self.space_pattern.sub(' ', text)
            text = text.strip()
        
        return text
    
    def extract_keywords(
        self,
        text: str,
        min_length: int = 2,
        max_length: int = 20,
        stop_words: Optional[Set[str]] = None
    ) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            min_length: Minimum keyword length
            max_length: Maximum keyword length
            stop_words: Set of stop words to exclude
            
        Returns:
            List of keywords
        """
        if not text:
            return []
        
        # Clean text first
        text = self.clean_text(text)
        
        # Tokenize into words
        # Handle both Korean and non-Korean text
        words = re.findall(
            r'[가-힣]+|[a-zA-Z]+|\d+',
            text
        )
        
        # Filter words
        keywords = []
        for word in words:
            word = word.strip()
            if (min_length <= len(word) <= max_length and
                (stop_words is None or word.lower() not in stop_words)):
                keywords.append(word)
        
        return keywords 