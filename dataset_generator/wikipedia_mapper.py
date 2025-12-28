"""
Wikipedia Passage Mapper for Cross-Entity Clusters

This module resolves Wikidata QIDs to Wikipedia articles, extracts passages
with section structure, and validates coverage for cluster entities.

Phase A.4: Wikipedia Passage Mapping
- A.4.1: Resolve QIDs to Wikipedia titles via Wikimapper
- A.4.2: Extract and parse Wikipedia articles with section structure
- A.4.3: Implement passage extraction with chunking strategy
- A.4.4: Add passage-level metadata (section, position, word count)
- A.4.5: Validate Wikipedia coverage per cluster
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import requests

from sparql_profiler import SPARQLProfiler, WIKIDATA_SPARQL_ENDPOINT


@dataclass
class WikipediaPassage:
    """A passage extracted from a Wikipedia article."""
    passage_id: str
    text: str
    section: str
    section_level: int
    position: int  # Position within article (0-indexed)
    char_start: int
    char_end: int
    word_count: int
    is_lead: bool
    metadata: dict = field(default_factory=dict)


@dataclass
class WikipediaArticle:
    """A Wikipedia article with extracted passages."""
    qid: str
    title: str
    url: str
    extract: str  # Lead paragraph
    sections: list
    passages: list
    total_word_count: int
    passage_count: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ClusterCoverage:
    """Wikipedia coverage statistics for a cluster."""
    cluster_id: str
    total_entities: int
    entities_with_wikipedia: int
    coverage_rate: float
    total_passages: int
    avg_passages_per_entity: float
    entities_without_wikipedia: list
    quality_distribution: dict


class WikimapperResolver:
    """
    Resolves Wikidata QIDs to Wikipedia titles.
    Uses SPARQL as primary method with caching.
    """

    def __init__(self, endpoint: str = WIKIDATA_SPARQL_ENDPOINT):
        self.profiler = SPARQLProfiler(endpoint)
        self.cache = {}

    def resolve_qid(self, qid: str) -> Optional[str]:
        """Resolve a single QID to Wikipedia title."""
        if qid in self.cache:
            return self.cache[qid]

        sparql = f"""
SELECT ?article WHERE {{
  ?article schema:about wd:{qid} .
  ?article schema:isPartOf <https://en.wikipedia.org/> .
}}
LIMIT 1
"""

        result = self.profiler.execute_query(sparql, timeout=30)

        if result.success and result.entities:
            url = result.entities[0].get("article", "")
            title = url.split("/wiki/")[-1].replace("_", " ") if url else None
            self.cache[qid] = title
            return title

        self.cache[qid] = None
        return None

    def batch_resolve(self, qids: list) -> dict:
        """
        Resolve multiple QIDs to Wikipedia titles in batch.

        Returns:
            Dictionary mapping QID to Wikipedia title (or None)
        """
        # Check cache first
        results = {}
        to_fetch = []

        for qid in qids:
            if qid in self.cache:
                results[qid] = self.cache[qid]
            else:
                to_fetch.append(qid)

        if not to_fetch:
            return results

        # Batch query
        batch_size = 100
        for i in range(0, len(to_fetch), batch_size):
            batch = to_fetch[i:i + batch_size]
            values = " ".join(f"wd:{qid}" for qid in batch)

            sparql = f"""
SELECT ?entity ?article WHERE {{
  VALUES ?entity {{ {values} }}
  ?article schema:about ?entity .
  ?article schema:isPartOf <https://en.wikipedia.org/> .
}}
"""

            result = self.profiler.execute_query(sparql, timeout=60)

            if result.success:
                for binding in result.entities:
                    qid = binding.get("entity", "").split("/")[-1]
                    url = binding.get("article", "")
                    title = url.split("/wiki/")[-1].replace("_", " ") if url else None
                    results[qid] = title
                    self.cache[qid] = title

            # Mark missing as None
            for qid in batch:
                if qid not in results:
                    results[qid] = None
                    self.cache[qid] = None

            time.sleep(0.5)  # Rate limiting

        return results


class WikipediaExtractor:
    """
    Extracts and parses Wikipedia articles with section structure.
    Uses Wikipedia API for article content.
    """

    WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
    HEADERS = {
        "User-Agent": "CrossEntityClusterBot/1.0 (HC-RAG Research Project; https://github.com/)"
    }

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def fetch_article(self, title: str) -> Optional[dict]:
        """
        Fetch article content from Wikipedia API.

        Returns:
            Dictionary with title, extract, sections, and full text
        """
        # Get article extract and sections
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts|sections",
            "exintro": True,
            "explaintext": True,
            "format": "json"
        }

        try:
            response = requests.get(
                self.WIKIPEDIA_API,
                params=params,
                headers=self.HEADERS,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return None

            page = list(pages.values())[0]
            if "missing" in page:
                return None

            extract = page.get("extract", "")
            sections_data = page.get("sections", [])

            # Get full article text
            full_params = {
                "action": "query",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
                "format": "json"
            }

            full_response = requests.get(
                self.WIKIPEDIA_API,
                params=full_params,
                headers=self.HEADERS,
                timeout=30
            )
            full_response.raise_for_status()
            full_data = full_response.json()
            full_pages = full_data.get("query", {}).get("pages", {})
            full_page = list(full_pages.values())[0]
            full_text = full_page.get("extract", "")

            return {
                "title": page.get("title", title),
                "pageid": page.get("pageid"),
                "extract": extract,
                "sections": sections_data,
                "full_text": full_text
            }

        except Exception as e:
            print(f"Error fetching {title}: {e}")
            return None

    def parse_sections(self, full_text: str, sections_data: list) -> list:
        """
        Parse article text into sections.

        Returns:
            List of (section_name, section_level, section_text) tuples
        """
        if not full_text:
            return []

        sections = []
        lines = full_text.split("\n")

        current_section = "Lead"
        current_level = 0
        current_text = []

        # Simple section detection based on headers
        section_pattern = re.compile(r'^(={2,})\s*(.+?)\s*\1$')

        for line in lines:
            match = section_pattern.match(line)
            if match:
                # Save previous section
                if current_text:
                    text = "\n".join(current_text).strip()
                    if text:
                        sections.append({
                            "name": current_section,
                            "level": current_level,
                            "text": text
                        })

                # Start new section
                level = len(match.group(1)) - 1  # == is level 1
                current_section = match.group(2).strip()
                current_level = level
                current_text = []
            else:
                current_text.append(line)

        # Save last section
        if current_text:
            text = "\n".join(current_text).strip()
            if text:
                sections.append({
                    "name": current_section,
                    "level": current_level,
                    "text": text
                })

        return sections

    def chunk_text(self, text: str, section_name: str, section_level: int) -> list:
        """
        Split text into chunks with overlap.

        Returns:
            List of chunk dictionaries
        """
        if not text:
            return []

        chunks = []
        words = text.split()

        if len(words) <= self.chunk_size // 4:  # Short text, don't chunk
            chunks.append({
                "text": text,
                "word_count": len(words),
                "section": section_name,
                "section_level": section_level
            })
            return chunks

        # Character-based chunking with word boundary respect
        start = 0
        while start < len(text):
            end = start + self.chunk_size

            # Find word boundary
            if end < len(text):
                # Look for space near the end
                while end > start and text[end] != ' ':
                    end -= 1
                if end == start:
                    end = start + self.chunk_size

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_words = chunk_text.split()
                chunks.append({
                    "text": chunk_text,
                    "word_count": len(chunk_words),
                    "char_start": start,
                    "char_end": end,
                    "section": section_name,
                    "section_level": section_level
                })

            start = end - self.chunk_overlap
            if start < 0:
                start = 0

        return chunks

    def extract_passages(self, qid: str, title: str) -> Optional[WikipediaArticle]:
        """
        Extract passages from a Wikipedia article.

        Returns:
            WikipediaArticle with all extracted passages
        """
        article_data = self.fetch_article(title)
        if not article_data:
            return None

        # Parse sections
        sections = self.parse_sections(
            article_data["full_text"],
            article_data.get("sections", [])
        )

        # Extract passages from each section
        passages = []
        position = 0

        for section in sections:
            is_lead = section["name"] == "Lead"
            chunks = self.chunk_text(
                section["text"],
                section["name"],
                section["level"]
            )

            for chunk in chunks:
                passage = WikipediaPassage(
                    passage_id=f"{qid}_p{position}",
                    text=chunk["text"],
                    section=chunk["section"],
                    section_level=chunk["section_level"],
                    position=position,
                    char_start=chunk.get("char_start", 0),
                    char_end=chunk.get("char_end", len(chunk["text"])),
                    word_count=chunk["word_count"],
                    is_lead=is_lead
                )
                passages.append(passage)
                position += 1

        # Calculate total word count
        total_words = sum(p.word_count for p in passages)

        return WikipediaArticle(
            qid=qid,
            title=article_data["title"],
            url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            extract=article_data["extract"],
            sections=[{"name": s["name"], "level": s["level"]} for s in sections],
            passages=passages,
            total_word_count=total_words,
            passage_count=len(passages),
            metadata={
                "pageid": article_data.get("pageid"),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }
        )


class ClusterWikipediaMapper:
    """
    Maps cluster entities to Wikipedia passages and validates coverage.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.resolver = WikimapperResolver()
        self.extractor = WikipediaExtractor(chunk_size, chunk_overlap)

    def map_cluster(
        self,
        cluster_data: dict,
        max_entities: Optional[int] = None,
        delay: float = 0.5
    ) -> tuple:
        """
        Map all entities in a cluster to Wikipedia passages.

        Args:
            cluster_data: Cluster dictionary with entities
            max_entities: Maximum entities to process (for testing)
            delay: Delay between API requests

        Returns:
            Tuple of (list of WikipediaArticle, ClusterCoverage)
        """
        entities = cluster_data.get("entities", [])
        cluster_id = cluster_data.get("cluster_id", "unknown")

        if max_entities:
            entities = entities[:max_entities]

        print(f"Mapping {len(entities)} entities to Wikipedia...")

        # Resolve QIDs to titles
        qids = [e.get("qid") for e in entities if e.get("qid")]
        titles = self.resolver.batch_resolve(qids)

        # Extract passages for each entity with Wikipedia
        articles = []
        entities_without_wiki = []

        for i, entity in enumerate(entities):
            qid = entity.get("qid")
            title = titles.get(qid)

            if not title:
                entities_without_wiki.append({
                    "qid": qid,
                    "label": entity.get("label", "")
                })
                continue

            print(f"  [{i+1}/{len(entities)}] Extracting: {title}")

            try:
                article = self.extractor.extract_passages(qid, title)
                if article:
                    articles.append(article)
                else:
                    entities_without_wiki.append({
                        "qid": qid,
                        "label": entity.get("label", ""),
                        "title": title,
                        "reason": "extraction_failed"
                    })

                time.sleep(delay)

            except Exception as e:
                print(f"    Error: {e}")
                entities_without_wiki.append({
                    "qid": qid,
                    "label": entity.get("label", ""),
                    "title": title,
                    "reason": str(e)
                })

        # Calculate coverage
        total_passages = sum(a.passage_count for a in articles)
        avg_passages = total_passages / len(articles) if articles else 0

        # Quality distribution
        quality_dist = {"high": 0, "medium": 0, "low": 0}
        for article in articles:
            if article.passage_count >= 10:
                quality_dist["high"] += 1
            elif article.passage_count >= 5:
                quality_dist["medium"] += 1
            else:
                quality_dist["low"] += 1

        coverage = ClusterCoverage(
            cluster_id=cluster_id,
            total_entities=len(entities),
            entities_with_wikipedia=len(articles),
            coverage_rate=len(articles) / len(entities) if entities else 0,
            total_passages=total_passages,
            avg_passages_per_entity=avg_passages,
            entities_without_wikipedia=entities_without_wiki,
            quality_distribution=quality_dist
        )

        return articles, coverage

    def validate_coverage(self, coverage: ClusterCoverage) -> dict:
        """
        Validate if cluster meets coverage requirements.

        Returns:
            Validation result dictionary
        """
        is_valid = (
            coverage.coverage_rate >= 0.8 and
            coverage.avg_passages_per_entity >= 3
        )

        issues = []
        if coverage.coverage_rate < 0.8:
            issues.append(f"Coverage rate {coverage.coverage_rate:.1%} below 80% threshold")
        if coverage.avg_passages_per_entity < 3:
            issues.append(f"Average passages {coverage.avg_passages_per_entity:.1f} below 3 minimum")

        recommendations = []
        if coverage.coverage_rate < 0.8:
            recommendations.append("Consider filtering entities without Wikipedia articles")
            recommendations.append("Check if alternative language Wikipedias have better coverage")
        if coverage.avg_passages_per_entity < 3:
            recommendations.append("Reduce chunk size to generate more passages")
            recommendations.append("Consider including stub articles with flag")

        return {
            "is_valid": is_valid,
            "issues": issues,
            "recommendations": recommendations,
            "summary": {
                "coverage_rate": f"{coverage.coverage_rate:.1%}",
                "entities_with_wiki": f"{coverage.entities_with_wikipedia}/{coverage.total_entities}",
                "total_passages": coverage.total_passages,
                "avg_passages": f"{coverage.avg_passages_per_entity:.1f}",
                "quality_distribution": coverage.quality_distribution
            }
        }


def article_to_dict(article: WikipediaArticle) -> dict:
    """Convert WikipediaArticle to dictionary."""
    return {
        "qid": article.qid,
        "title": article.title,
        "url": article.url,
        "extract": article.extract,
        "sections": article.sections,
        "passages": [
            {
                "passage_id": p.passage_id,
                "text": p.text,
                "section": p.section,
                "section_level": p.section_level,
                "position": p.position,
                "char_start": p.char_start,
                "char_end": p.char_end,
                "word_count": p.word_count,
                "is_lead": p.is_lead
            }
            for p in article.passages
        ],
        "total_word_count": article.total_word_count,
        "passage_count": article.passage_count,
        "metadata": article.metadata
    }


def coverage_to_dict(coverage: ClusterCoverage) -> dict:
    """Convert ClusterCoverage to dictionary."""
    return {
        "cluster_id": coverage.cluster_id,
        "total_entities": coverage.total_entities,
        "entities_with_wikipedia": coverage.entities_with_wikipedia,
        "coverage_rate": coverage.coverage_rate,
        "total_passages": coverage.total_passages,
        "avg_passages_per_entity": coverage.avg_passages_per_entity,
        "entities_without_wikipedia": coverage.entities_without_wikipedia,
        "quality_distribution": coverage.quality_distribution
    }


def save_cluster_passages(
    cluster_id: str,
    articles: list,
    coverage: ClusterCoverage,
    output_dir: str
):
    """Save cluster passages and coverage to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save passages
    passages_file = output_path / f"{cluster_id}_passages.json"
    with open(passages_file, "w", encoding="utf-8") as f:
        json.dump({
            "cluster_id": cluster_id,
            "article_count": len(articles),
            "articles": [article_to_dict(a) for a in articles]
        }, f, indent=2, ensure_ascii=False)

    # Save coverage
    coverage_file = output_path / f"{cluster_id}_coverage.json"
    with open(coverage_file, "w", encoding="utf-8") as f:
        json.dump(coverage_to_dict(coverage), f, indent=2, ensure_ascii=False)

    print(f"Saved {len(articles)} articles to {passages_file}")
    print(f"Saved coverage report to {coverage_file}")

    return passages_file, coverage_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Map cluster entities to Wikipedia passages")
    parser.add_argument("--cluster", type=str, required=True, help="Path to cluster JSON file")
    parser.add_argument("--output", type=str, default="data/passages", help="Output directory")
    parser.add_argument("--max-entities", type=int, help="Maximum entities to process")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap in characters")

    args = parser.parse_args()

    # Load cluster
    with open(args.cluster, "r", encoding="utf-8") as f:
        cluster_data = json.load(f)

    cluster_id = cluster_data.get("cluster_id", "unknown")
    print(f"Processing cluster: {cluster_id}")

    # Map to Wikipedia
    mapper = ClusterWikipediaMapper(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    articles, coverage = mapper.map_cluster(
        cluster_data,
        max_entities=args.max_entities
    )

    # Validate coverage
    validation = mapper.validate_coverage(coverage)

    # Print summary
    print("\n" + "=" * 60)
    print("COVERAGE SUMMARY")
    print("=" * 60)
    print(f"Entities with Wikipedia: {coverage.entities_with_wikipedia}/{coverage.total_entities}")
    print(f"Coverage rate: {coverage.coverage_rate:.1%}")
    print(f"Total passages: {coverage.total_passages}")
    print(f"Average passages per entity: {coverage.avg_passages_per_entity:.1f}")
    print(f"Quality distribution: {coverage.quality_distribution}")

    print(f"\nValidation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    for issue in validation["issues"]:
        print(f"  - {issue}")

    # Save results
    save_cluster_passages(cluster_id, articles, coverage, args.output)
