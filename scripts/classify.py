ent_path = "../FB15k/train.entlist"
rel_path = "../FB15k/train.rellist"
import_path = "../FB15k/whole.txt"
export_path1 = "../FB15k/FB15k_head_classified.txt"
export_path2 = "../FB15k/FB15k_tail_classified.txt"

with open(ent_path, "r") as ent_path, open(rel_path, "r") as rel_path, open (import_path, "r") as in_file ,open(export_path1, "w") as out_file1, open(export_path2, "w") as out_file2:
	entlist = []
	rellist = []
	head2rel = {}
	tail2rel = {}
	print("loading relations")
	for line in rel_path:
		rel = line.strip()
		rellist.append(rel)

	print("loading entities")
	for line in ent_path:
		ent = line.strip()
		entlist.append(ent)
		head2rel[ent] = {}
		tail2rel[ent] = {}
		for rel in rellist:
			head2rel[ent][rel] = 0
			tail2rel[ent][rel] = 0

	print("loading triplets")
	for line in in_file:
		head, rel, tail = line.strip().split('\t')
		head2rel[head][rel] = head2rel[head][rel] + 1

		tail2rel[tail][rel] = tail2rel[tail][rel] + 1

	print("printing results")
	print("entities in triplets as head", file=out_file1)
	cnt = 0
	for ent in entlist:
		cnt = 0
		print(ent, head2rel[ent].values(), file=out_file1)
	print("entities in triplets as tail", file=out_file2)
	for ent in entlist:
		print(ent, tail2rel[ent].values(), file=out_file2)
