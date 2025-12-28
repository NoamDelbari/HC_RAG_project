from entity_enumerator import EntityEnumerator, save_cluster
e = EntityEnumerator()
cluster = e.enumerate_position_holders('Q144478', 'UN Secretary-General')
print(f'Found {cluster.entity_count} entities')
for ent in cluster.entities[:5]:
    print(f'  {ent.qid}: {ent.label}')