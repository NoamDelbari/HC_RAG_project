from wikipedia_mapper import WikimapperResolver, WikipediaExtractor

# Test QID resolution
resolver = WikimapperResolver()
title = resolver.resolve_qid('Q9916')  # Albert Einstein
print(f'Wikipedia title: {title}')

# Test passage extraction
extractor = WikipediaExtractor(chunk_size=800)
article = extractor.extract_passages('Q9916', title)
if article:
    print(f'Passages: {article.passage_count}')
    print(f'Sections: {[s["name"] for s in article.sections[:5]]}')