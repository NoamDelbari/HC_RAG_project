"""
Entity Metadata Enricher

This module enriches cluster entities with additional metadata including
temporal, geographic, and relational properties from Wikidata.

Phase A.3.2: Enrich entities with metadata (temporal, geographic, relational)
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from sparql_profiler import SPARQLProfiler, WIKIDATA_SPARQL_ENDPOINT
from entity_enumerator import Entity, ClusterEntities, cluster_to_dict


@dataclass
class TemporalMetadata:
    """Temporal properties of an entity."""
    birth_date: Optional[str] = None
    death_date: Optional[str] = None
    inception_date: Optional[str] = None
    dissolution_date: Optional[str] = None
    publication_date: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class GeographicMetadata:
    """Geographic properties of an entity."""
    country: Optional[str] = None
    country_qid: Optional[str] = None
    birth_place: Optional[str] = None
    birth_place_qid: Optional[str] = None
    death_place: Optional[str] = None
    death_place_qid: Optional[str] = None
    location: Optional[str] = None
    location_qid: Optional[str] = None
    headquarters: Optional[str] = None
    headquarters_qid: Optional[str] = None
    coordinates: Optional[tuple] = None


@dataclass
class RelationalMetadata:
    """Relational properties of an entity."""
    instance_of: list = field(default_factory=list)
    occupation: list = field(default_factory=list)
    field_of_work: list = field(default_factory=list)
    employer: list = field(default_factory=list)
    educated_at: list = field(default_factory=list)
    awards: list = field(default_factory=list)
    spouse: list = field(default_factory=list)
    children: list = field(default_factory=list)


@dataclass
class EnrichedEntity:
    """Entity with full metadata."""
    qid: str
    label: str
    description: str
    aliases: list
    wikipedia_title: Optional[str]
    temporal: TemporalMetadata
    geographic: GeographicMetadata
    relational: RelationalMetadata
    raw_properties: dict = field(default_factory=dict)


class EntityEnricher:
    """
    Enriches entities with additional metadata from Wikidata.
    """

    def __init__(self, endpoint: str = WIKIDATA_SPARQL_ENDPOINT):
        self.profiler = SPARQLProfiler(endpoint)

    def enrich_person(self, qid: str) -> EnrichedEntity:
        """
        Enrich a person entity with full metadata.
        """
        sparql = f"""
SELECT ?label ?description ?birthDate ?deathDate
       ?birthPlaceLabel ?birthPlace ?deathPlaceLabel ?deathPlace
       ?countryLabel ?country
       ?occupationLabel ?occupation
       ?fieldLabel ?field
       ?educatedAtLabel ?educatedAt
       ?awardLabel ?award
       ?employerLabel ?employer
       ?spouseLabel ?spouse
       ?wikipedia
WHERE {{
  BIND(wd:{qid} AS ?entity)

  # Basic info
  OPTIONAL {{ ?entity rdfs:label ?label . FILTER(LANG(?label) = "en") }}
  OPTIONAL {{ ?entity schema:description ?description . FILTER(LANG(?description) = "en") }}

  # Temporal
  OPTIONAL {{ ?entity wdt:P569 ?birthDate }}
  OPTIONAL {{ ?entity wdt:P570 ?deathDate }}

  # Geographic
  OPTIONAL {{ ?entity wdt:P19 ?birthPlace . ?birthPlace rdfs:label ?birthPlaceLabel . FILTER(LANG(?birthPlaceLabel) = "en") }}
  OPTIONAL {{ ?entity wdt:P20 ?deathPlace . ?deathPlace rdfs:label ?deathPlaceLabel . FILTER(LANG(?deathPlaceLabel) = "en") }}
  OPTIONAL {{ ?entity wdt:P27 ?country . ?country rdfs:label ?countryLabel . FILTER(LANG(?countryLabel) = "en") }}

  # Relational
  OPTIONAL {{ ?entity wdt:P106 ?occupation . ?occupation rdfs:label ?occupationLabel . FILTER(LANG(?occupationLabel) = "en") }}
  OPTIONAL {{ ?entity wdt:P101 ?field . ?field rdfs:label ?fieldLabel . FILTER(LANG(?fieldLabel) = "en") }}
  OPTIONAL {{ ?entity wdt:P69 ?educatedAt . ?educatedAt rdfs:label ?educatedAtLabel . FILTER(LANG(?educatedAtLabel) = "en") }}
  OPTIONAL {{ ?entity wdt:P166 ?award . ?award rdfs:label ?awardLabel . FILTER(LANG(?awardLabel) = "en") }}
  OPTIONAL {{ ?entity wdt:P108 ?employer . ?employer rdfs:label ?employerLabel . FILTER(LANG(?employerLabel) = "en") }}
  OPTIONAL {{ ?entity wdt:P26 ?spouse . ?spouse rdfs:label ?spouseLabel . FILTER(LANG(?spouseLabel) = "en") }}

  # Wikipedia
  OPTIONAL {{
    ?wikipedia schema:about ?entity .
    ?wikipedia schema:isPartOf <https://en.wikipedia.org/> .
  }}
}}
LIMIT 100
"""

        result = self.profiler.execute_query(sparql, timeout=60)

        if not result.success or not result.entities:
            return None

        # Aggregate results (multiple rows for multi-valued properties)
        temporal = TemporalMetadata()
        geographic = GeographicMetadata()
        relational = RelationalMetadata()

        label = ""
        description = ""
        wikipedia_title = None

        occupations = set()
        fields = set()
        education = set()
        awards = set()
        employers = set()
        spouses = set()

        for binding in result.entities:
            # Single-valued
            if not label and binding.get("label"):
                label = binding["label"]
            if not description and binding.get("description"):
                description = binding["description"]
            if not wikipedia_title and binding.get("wikipedia"):
                wiki_url = binding["wikipedia"]
                wikipedia_title = wiki_url.split("/wiki/")[-1].replace("_", " ")

            # Temporal
            if not temporal.birth_date and binding.get("birthDate"):
                temporal.birth_date = binding["birthDate"]
            if not temporal.death_date and binding.get("deathDate"):
                temporal.death_date = binding["deathDate"]

            # Geographic
            if not geographic.birth_place and binding.get("birthPlaceLabel"):
                geographic.birth_place = binding["birthPlaceLabel"]
                geographic.birth_place_qid = binding.get("birthPlace", "").split("/")[-1]
            if not geographic.death_place and binding.get("deathPlaceLabel"):
                geographic.death_place = binding["deathPlaceLabel"]
                geographic.death_place_qid = binding.get("deathPlace", "").split("/")[-1]
            if not geographic.country and binding.get("countryLabel"):
                geographic.country = binding["countryLabel"]
                geographic.country_qid = binding.get("country", "").split("/")[-1]

            # Multi-valued relational
            if binding.get("occupationLabel"):
                occupations.add((binding["occupationLabel"], binding.get("occupation", "").split("/")[-1]))
            if binding.get("fieldLabel"):
                fields.add((binding["fieldLabel"], binding.get("field", "").split("/")[-1]))
            if binding.get("educatedAtLabel"):
                education.add((binding["educatedAtLabel"], binding.get("educatedAt", "").split("/")[-1]))
            if binding.get("awardLabel"):
                awards.add((binding["awardLabel"], binding.get("award", "").split("/")[-1]))
            if binding.get("employerLabel"):
                employers.add((binding["employerLabel"], binding.get("employer", "").split("/")[-1]))
            if binding.get("spouseLabel"):
                spouses.add((binding["spouseLabel"], binding.get("spouse", "").split("/")[-1]))

        # Convert sets to lists of dicts
        relational.occupation = [{"label": l, "qid": q} for l, q in occupations]
        relational.field_of_work = [{"label": l, "qid": q} for l, q in fields]
        relational.educated_at = [{"label": l, "qid": q} for l, q in education]
        relational.awards = [{"label": l, "qid": q} for l, q in awards]
        relational.employer = [{"label": l, "qid": q} for l, q in employers]
        relational.spouse = [{"label": l, "qid": q} for l, q in spouses]

        return EnrichedEntity(
            qid=qid,
            label=label,
            description=description,
            aliases=[],
            wikipedia_title=wikipedia_title,
            temporal=temporal,
            geographic=geographic,
            relational=relational
        )

    def enrich_creative_work(self, qid: str) -> EnrichedEntity:
        """
        Enrich a creative work (film, book, album) with metadata.
        """
        sparql = f"""
SELECT ?label ?description ?pubDate ?country ?countryLabel
       ?director ?directorLabel ?genre ?genreLabel
       ?producer ?producerLabel ?duration
       ?wikipedia
WHERE {{
  BIND(wd:{qid} AS ?entity)

  OPTIONAL {{ ?entity rdfs:label ?label . FILTER(LANG(?label) = "en") }}
  OPTIONAL {{ ?entity schema:description ?description . FILTER(LANG(?description) = "en") }}

  # Temporal
  OPTIONAL {{ ?entity wdt:P577 ?pubDate }}

  # Geographic
  OPTIONAL {{ ?entity wdt:P495 ?country . ?country rdfs:label ?countryLabel . FILTER(LANG(?countryLabel) = "en") }}

  # Relational
  OPTIONAL {{ ?entity wdt:P57 ?director . ?director rdfs:label ?directorLabel . FILTER(LANG(?directorLabel) = "en") }}
  OPTIONAL {{ ?entity wdt:P136 ?genre . ?genre rdfs:label ?genreLabel . FILTER(LANG(?genreLabel) = "en") }}
  OPTIONAL {{ ?entity wdt:P162 ?producer . ?producer rdfs:label ?producerLabel . FILTER(LANG(?producerLabel) = "en") }}
  OPTIONAL {{ ?entity wdt:P2047 ?duration }}

  # Wikipedia
  OPTIONAL {{
    ?wikipedia schema:about ?entity .
    ?wikipedia schema:isPartOf <https://en.wikipedia.org/> .
  }}
}}
LIMIT 50
"""

        result = self.profiler.execute_query(sparql, timeout=60)

        if not result.success or not result.entities:
            return None

        temporal = TemporalMetadata()
        geographic = GeographicMetadata()
        relational = RelationalMetadata()

        label = ""
        description = ""
        wikipedia_title = None
        raw_properties = {}

        directors = set()
        genres = set()
        producers = set()

        for binding in result.entities:
            if not label and binding.get("label"):
                label = binding["label"]
            if not description and binding.get("description"):
                description = binding["description"]
            if not wikipedia_title and binding.get("wikipedia"):
                wiki_url = binding["wikipedia"]
                wikipedia_title = wiki_url.split("/wiki/")[-1].replace("_", " ")

            if not temporal.publication_date and binding.get("pubDate"):
                temporal.publication_date = binding["pubDate"]

            if not geographic.country and binding.get("countryLabel"):
                geographic.country = binding["countryLabel"]
                geographic.country_qid = binding.get("country", "").split("/")[-1]

            if binding.get("directorLabel"):
                directors.add((binding["directorLabel"], binding.get("director", "").split("/")[-1]))
            if binding.get("genreLabel"):
                genres.add((binding["genreLabel"], binding.get("genre", "").split("/")[-1]))
            if binding.get("producerLabel"):
                producers.add((binding["producerLabel"], binding.get("producer", "").split("/")[-1]))

            if binding.get("duration"):
                raw_properties["duration_minutes"] = binding["duration"]

        raw_properties["directors"] = [{"label": l, "qid": q} for l, q in directors]
        raw_properties["genres"] = [{"label": l, "qid": q} for l, q in genres]
        raw_properties["producers"] = [{"label": l, "qid": q} for l, q in producers]

        return EnrichedEntity(
            qid=qid,
            label=label,
            description=description,
            aliases=[],
            wikipedia_title=wikipedia_title,
            temporal=temporal,
            geographic=geographic,
            relational=relational,
            raw_properties=raw_properties
        )

    def enrich_organization(self, qid: str) -> EnrichedEntity:
        """
        Enrich an organization (company, university, etc) with metadata.
        """
        sparql = f"""
SELECT ?label ?description ?inception ?dissolution
       ?country ?countryLabel ?hq ?hqLabel
       ?founder ?founderLabel ?industry ?industryLabel
       ?employees ?wikipedia
WHERE {{
  BIND(wd:{qid} AS ?entity)

  OPTIONAL {{ ?entity rdfs:label ?label . FILTER(LANG(?label) = "en") }}
  OPTIONAL {{ ?entity schema:description ?description . FILTER(LANG(?description) = "en") }}

  # Temporal
  OPTIONAL {{ ?entity wdt:P571 ?inception }}
  OPTIONAL {{ ?entity wdt:P576 ?dissolution }}

  # Geographic
  OPTIONAL {{ ?entity wdt:P17 ?country . ?country rdfs:label ?countryLabel . FILTER(LANG(?countryLabel) = "en") }}
  OPTIONAL {{ ?entity wdt:P159 ?hq . ?hq rdfs:label ?hqLabel . FILTER(LANG(?hqLabel) = "en") }}

  # Relational
  OPTIONAL {{ ?entity wdt:P112 ?founder . ?founder rdfs:label ?founderLabel . FILTER(LANG(?founderLabel) = "en") }}
  OPTIONAL {{ ?entity wdt:P452 ?industry . ?industry rdfs:label ?industryLabel . FILTER(LANG(?industryLabel) = "en") }}

  # Numeric
  OPTIONAL {{ ?entity wdt:P1128 ?employees }}

  # Wikipedia
  OPTIONAL {{
    ?wikipedia schema:about ?entity .
    ?wikipedia schema:isPartOf <https://en.wikipedia.org/> .
  }}
}}
LIMIT 50
"""

        result = self.profiler.execute_query(sparql, timeout=60)

        if not result.success or not result.entities:
            return None

        temporal = TemporalMetadata()
        geographic = GeographicMetadata()
        relational = RelationalMetadata()

        label = ""
        description = ""
        wikipedia_title = None
        raw_properties = {}

        founders = set()
        industries = set()

        for binding in result.entities:
            if not label and binding.get("label"):
                label = binding["label"]
            if not description and binding.get("description"):
                description = binding["description"]
            if not wikipedia_title and binding.get("wikipedia"):
                wiki_url = binding["wikipedia"]
                wikipedia_title = wiki_url.split("/wiki/")[-1].replace("_", " ")

            if not temporal.inception_date and binding.get("inception"):
                temporal.inception_date = binding["inception"]
            if not temporal.dissolution_date and binding.get("dissolution"):
                temporal.dissolution_date = binding["dissolution"]

            if not geographic.country and binding.get("countryLabel"):
                geographic.country = binding["countryLabel"]
                geographic.country_qid = binding.get("country", "").split("/")[-1]
            if not geographic.headquarters and binding.get("hqLabel"):
                geographic.headquarters = binding["hqLabel"]
                geographic.headquarters_qid = binding.get("hq", "").split("/")[-1]

            if binding.get("founderLabel"):
                founders.add((binding["founderLabel"], binding.get("founder", "").split("/")[-1]))
            if binding.get("industryLabel"):
                industries.add((binding["industryLabel"], binding.get("industry", "").split("/")[-1]))

            if binding.get("employees"):
                raw_properties["employees"] = binding["employees"]

        raw_properties["founders"] = [{"label": l, "qid": q} for l, q in founders]
        raw_properties["industries"] = [{"label": l, "qid": q} for l, q in industries]

        return EnrichedEntity(
            qid=qid,
            label=label,
            description=description,
            aliases=[],
            wikipedia_title=wikipedia_title,
            temporal=temporal,
            geographic=geographic,
            relational=relational,
            raw_properties=raw_properties
        )

    def enrich_location(self, qid: str) -> EnrichedEntity:
        """
        Enrich a location (city, country, etc) with metadata.
        """
        sparql = f"""
SELECT ?label ?description ?country ?countryLabel
       ?adminRegion ?adminRegionLabel ?population
       ?coordinates ?capital ?capitalLabel
       ?wikipedia
WHERE {{
  BIND(wd:{qid} AS ?entity)

  OPTIONAL {{ ?entity rdfs:label ?label . FILTER(LANG(?label) = "en") }}
  OPTIONAL {{ ?entity schema:description ?description . FILTER(LANG(?description) = "en") }}

  # Geographic
  OPTIONAL {{ ?entity wdt:P17 ?country . ?country rdfs:label ?countryLabel . FILTER(LANG(?countryLabel) = "en") }}
  OPTIONAL {{ ?entity wdt:P131 ?adminRegion . ?adminRegion rdfs:label ?adminRegionLabel . FILTER(LANG(?adminRegionLabel) = "en") }}
  OPTIONAL {{ ?entity wdt:P625 ?coordinates }}
  OPTIONAL {{ ?entity wdt:P36 ?capital . ?capital rdfs:label ?capitalLabel . FILTER(LANG(?capitalLabel) = "en") }}

  # Numeric
  OPTIONAL {{ ?entity wdt:P1082 ?population }}

  # Wikipedia
  OPTIONAL {{
    ?wikipedia schema:about ?entity .
    ?wikipedia schema:isPartOf <https://en.wikipedia.org/> .
  }}
}}
LIMIT 20
"""

        result = self.profiler.execute_query(sparql, timeout=60)

        if not result.success or not result.entities:
            return None

        temporal = TemporalMetadata()
        geographic = GeographicMetadata()
        relational = RelationalMetadata()

        label = ""
        description = ""
        wikipedia_title = None
        raw_properties = {}

        for binding in result.entities:
            if not label and binding.get("label"):
                label = binding["label"]
            if not description and binding.get("description"):
                description = binding["description"]
            if not wikipedia_title and binding.get("wikipedia"):
                wiki_url = binding["wikipedia"]
                wikipedia_title = wiki_url.split("/wiki/")[-1].replace("_", " ")

            if not geographic.country and binding.get("countryLabel"):
                geographic.country = binding["countryLabel"]
                geographic.country_qid = binding.get("country", "").split("/")[-1]
            if not geographic.location and binding.get("adminRegionLabel"):
                geographic.location = binding["adminRegionLabel"]
                geographic.location_qid = binding.get("adminRegion", "").split("/")[-1]

            if binding.get("coordinates"):
                raw_properties["coordinates"] = binding["coordinates"]
            if binding.get("population"):
                raw_properties["population"] = binding["population"]
            if binding.get("capitalLabel"):
                raw_properties["capital"] = {
                    "label": binding["capitalLabel"],
                    "qid": binding.get("capital", "").split("/")[-1]
                }

        return EnrichedEntity(
            qid=qid,
            label=label,
            description=description,
            aliases=[],
            wikipedia_title=wikipedia_title,
            temporal=temporal,
            geographic=geographic,
            relational=relational,
            raw_properties=raw_properties
        )

    def batch_enrich_entities(
        self,
        qids: list,
        entity_type: str = "person",
        delay: float = 0.5
    ) -> list:
        """
        Enrich multiple entities in batch.

        Args:
            qids: List of Wikidata QIDs
            entity_type: Type of entities (person, creative_work, organization, location)
            delay: Delay between requests in seconds

        Returns:
            List of EnrichedEntity objects
        """
        enricher_map = {
            "person": self.enrich_person,
            "creative_work": self.enrich_creative_work,
            "organization": self.enrich_organization,
            "location": self.enrich_location
        }

        enricher = enricher_map.get(entity_type, self.enrich_person)
        enriched = []

        for i, qid in enumerate(qids):
            try:
                entity = enricher(qid)
                if entity:
                    enriched.append(entity)
                    print(f"  [{i+1}/{len(qids)}] Enriched {qid}: {entity.label}")
                else:
                    print(f"  [{i+1}/{len(qids)}] Failed to enrich {qid}")

                time.sleep(delay)

            except Exception as e:
                print(f"  [{i+1}/{len(qids)}] Error enriching {qid}: {e}")

        return enriched

    def enrich_cluster(
        self,
        cluster: ClusterEntities,
        entity_type: str = "person"
    ) -> ClusterEntities:
        """
        Enrich all entities in a cluster.
        """
        print(f"Enriching cluster: {cluster.cluster_name} ({cluster.entity_count} entities)")

        qids = [e.qid for e in cluster.entities]
        enriched = self.batch_enrich_entities(qids, entity_type)

        # Create mapping from QID to enriched entity
        enriched_map = {e.qid: e for e in enriched}

        # Update cluster entities
        for entity in cluster.entities:
            if entity.qid in enriched_map:
                enriched_entity = enriched_map[entity.qid]

                # Update entity properties
                entity.description = enriched_entity.description or entity.description
                entity.wikipedia_title = enriched_entity.wikipedia_title or entity.wikipedia_title

                # Add enriched metadata to properties
                entity.properties["temporal"] = {
                    "birth_date": enriched_entity.temporal.birth_date,
                    "death_date": enriched_entity.temporal.death_date,
                    "inception_date": enriched_entity.temporal.inception_date,
                    "publication_date": enriched_entity.temporal.publication_date
                }
                entity.properties["geographic"] = {
                    "country": enriched_entity.geographic.country,
                    "country_qid": enriched_entity.geographic.country_qid,
                    "birth_place": enriched_entity.geographic.birth_place,
                    "location": enriched_entity.geographic.location
                }
                entity.properties["relational"] = {
                    "occupation": enriched_entity.relational.occupation,
                    "field_of_work": enriched_entity.relational.field_of_work,
                    "educated_at": enriched_entity.relational.educated_at,
                    "awards": enriched_entity.relational.awards[:5]  # Limit to avoid bloat
                }

        cluster.metadata["enriched"] = True
        cluster.metadata["enrichment_coverage"] = len(enriched) / len(qids) if qids else 0

        return cluster


def enriched_entity_to_dict(entity: EnrichedEntity) -> dict:
    """Convert EnrichedEntity to dictionary."""
    return {
        "qid": entity.qid,
        "label": entity.label,
        "description": entity.description,
        "aliases": entity.aliases,
        "wikipedia_title": entity.wikipedia_title,
        "temporal": {
            "birth_date": entity.temporal.birth_date,
            "death_date": entity.temporal.death_date,
            "inception_date": entity.temporal.inception_date,
            "dissolution_date": entity.temporal.dissolution_date,
            "publication_date": entity.temporal.publication_date,
            "start_date": entity.temporal.start_date,
            "end_date": entity.temporal.end_date
        },
        "geographic": {
            "country": entity.geographic.country,
            "country_qid": entity.geographic.country_qid,
            "birth_place": entity.geographic.birth_place,
            "birth_place_qid": entity.geographic.birth_place_qid,
            "death_place": entity.geographic.death_place,
            "location": entity.geographic.location,
            "headquarters": entity.geographic.headquarters
        },
        "relational": {
            "occupation": entity.relational.occupation,
            "field_of_work": entity.relational.field_of_work,
            "educated_at": entity.relational.educated_at,
            "awards": entity.relational.awards,
            "employer": entity.relational.employer,
            "spouse": entity.relational.spouse
        },
        "raw_properties": entity.raw_properties
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enrich entities with metadata")
    parser.add_argument("--qid", type=str, help="Single QID to enrich")
    parser.add_argument("--type", type=str, default="person",
                        choices=["person", "creative_work", "organization", "location"])
    parser.add_argument("--cluster", type=str, help="Path to cluster JSON file to enrich")
    parser.add_argument("--output", type=str, help="Output file path")

    args = parser.parse_args()

    enricher = EntityEnricher()

    if args.qid:
        # Enrich single entity
        print(f"Enriching {args.qid} as {args.type}...")

        if args.type == "person":
            entity = enricher.enrich_person(args.qid)
        elif args.type == "creative_work":
            entity = enricher.enrich_creative_work(args.qid)
        elif args.type == "organization":
            entity = enricher.enrich_organization(args.qid)
        else:
            entity = enricher.enrich_location(args.qid)

        if entity:
            print(json.dumps(enriched_entity_to_dict(entity), indent=2))
        else:
            print("Failed to enrich entity")

    elif args.cluster:
        # Enrich cluster
        print(f"Loading cluster from {args.cluster}...")

        with open(args.cluster, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct cluster
        entities = [
            Entity(
                qid=e["qid"],
                label=e["label"],
                description=e.get("description", ""),
                wikipedia_title=e.get("wikipedia_title"),
                properties=e.get("properties", {})
            )
            for e in data["entities"]
        ]

        cluster = ClusterEntities(
            cluster_id=data["cluster_id"],
            cluster_name=data["cluster_name"],
            cluster_type=data["cluster_type"],
            description=data["description"],
            entity_count=data["entity_count"],
            entities=entities,
            binding_property=data.get("binding_property", ""),
            binding_value=data.get("binding_value", ""),
            constraints=data.get("constraints", []),
            metadata=data.get("metadata", {})
        )

        # Enrich
        enriched_cluster = enricher.enrich_cluster(cluster, entity_type=args.type)

        # Save
        output_path = args.output or args.cluster.replace(".json", "_enriched.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cluster_to_dict(enriched_cluster), f, indent=2, ensure_ascii=False)

        print(f"Enriched cluster saved to {output_path}")
