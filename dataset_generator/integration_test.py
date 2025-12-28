from entity_enumerator import EntityEnumerator, save_cluster
e = EntityEnumerator()
cluster = e.enumerate_series_members('Q8337', 'Harry Potter')
save_cluster(cluster, 'data/clusters')