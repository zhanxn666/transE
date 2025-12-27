from numpy import *
import operator

class Test:
    def __init__(self, entityList, entityVectorList, relationList ,relationVectorList, tripleListTrain, tripleListTest, label = "head", isFit = False):
        self.entityList = {}
        self.relationList = {}
        for name, vec in zip(entityList, entityVectorList):
            self.entityList[name] = vec
        for name, vec in zip(relationList, relationVectorList):
            self.relationList[name] = vec
        self.tripleListTrain = tripleListTrain
        self.tripleListTest = tripleListTest
        self.rank = []
        self.label = label
        self.isFit = isFit
        # After loading embeddings and triples
        print(f"Loaded {len(self.entityList)} entities and {len(self.relationList)} relations")

        # Check how many unique entities/relations in test set are missing
        test_entities = set()
        test_relations = set()
        for h, r, t in self.tripleListTest:
            test_entities.add(h)
            test_entities.add(t)
            test_relations.add(r)
        missing_entities = test_entities - set(self.entityList)
        missing_relations = test_relations - set(self.relationList)

        print(f"Missing entities in test set: {len(missing_entities)} / {len(test_entities)}")
        print(f"Missing relations in test set: {len(missing_relations)} / {len(test_relations)}")

        if missing_entities:
            print("Example missing entities:", list(missing_entities)[:10])
    def writeRank(self, dir):
        print("写入")
        file = open(dir, 'w')
        for r in self.rank:
            file.write(str(r[0]) + "\t")
            file.write(str(r[1]) + "\t")
            file.write(str(r[2]) + "\t")
            file.write(str(r[3]) + "\n")
        file.close()

    def getRank(self):
        print(f"Starting {self.label} prediction (filtered={self.isFit}) on {len(self.tripleListTest)} test triples...")
        cou = 0
        skipped = 0

        for triplet in self.tripleListTest:
            h, r, t = triplet  # head, relation, tail

            # Safety check: skip if any component missing in embeddings
            if h not in self.entityList or t not in self.entityList or r not in self.relationList:
                skipped += 1
                cou += 1
                if skipped <= 10:  # print first few warnings
                    print(f"Skipping triple {triplet} - missing embedding")
                continue

            rankList = {}

            for entityTemp in self.entityList.keys():
                if self.label == "head":
                    corruptedTriplet = (entityTemp, r, t)
                    if self.isFit and corruptedTriplet in self.tripleListTrain:
                        continue
                    # Correct: head_candidate + relation - tail
                    dist = distance(self.entityList[entityTemp], self.entityList[t], self.relationList[r])
                else:  # tail prediction
                    corruptedTriplet = (h, entityTemp, t)
                    if self.isFit and corruptedTriplet in self.tripleListTrain:
                        continue
                    # Correct: head + relation - tail_candidate
                    dist = distance(self.entityList[h], self.entityList[entityTemp], self.relationList[r])

                rankList[entityTemp] = dist

            # Sort by distance (ascending)
            nameRank = sorted(rankList.items(), key=operator.itemgetter(1))

            # True entity to rank
            true_entity = h if self.label == "head" else t

            # Compute rank
            rank = 1
            for candidate_entity, _ in nameRank:
                if candidate_entity == true_entity:
                    break
                rank += 1

            # Store result
            best_pred = nameRank[0][0]  # top-1 prediction
            self.rank.append((triplet, true_entity, best_pred, rank))

            print(rank)  # current triple's rank

            cou += 1
            if cou % 10000 == 0:
                print(f"Processed {cou} triples")

        print(f"Finished {self.label} prediction. "
            f"Processed: {cou - skipped}, Skipped (missing embeddings): {skipped}")

    def getRelationRank(self):
        cou = 0
        self.rank = []
        for triplet in self.tripleListTest:
            rankList = {}
            for relationTemp in self.relationList.keys():
                corruptedTriplet = (triplet[0], triplet[1], relationTemp)
                if self.isFit and (corruptedTriplet in self.tripleListTrain):
                    continue
                rankList[relationTemp] = distance(self.entityList[triplet[0]], self.entityList[triplet[1]], self.relationList[relationTemp])
            nameRank = sorted(rankList.items(), key = operator.itemgetter(1))
            x = 1
            for i in nameRank:
                if i[0] == triplet[2]:
                    break
                x += 1
            self.rank.append((triplet, triplet[2], nameRank[0][0], x))
            print(x)
            cou += 1
            if cou % 10000 == 0:
                print(cou)

    def getMeanRank(self):
        num = 0
        for r in self.rank:
            num += r[3]
        return num/len(self.rank)


def distance(h, t, r):
    h = array(h)
    t = array(t)
    r = array(r)
    s = h + r - t
    return linalg.norm(s)

def openD(dir, sp="\t"):
    #triple = (head, tail, relation)
    num = 0
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            triple = line.strip().split(sp)
            if(len(triple)<3):
                continue
            list.append(tuple(triple))
            num += 1
    print(num)
    return num, list

def loadData(str):
    fr = open(str)
    sArr = [line.strip().split("\t") for line in fr.readlines()]
    datArr = [[float(s) for s in line[1][1:-1].split(", ")] for line in sArr]
    nameArr = [line[0] for line in sArr]
    return datArr, nameArr

if __name__ == '__main__':
    dirTrain = "/home/e706/zhanxiangning/Tianchi/KnowledgeGraph/OpenBG500/OpenBG500_train.tsv"
    tripleNumTrain, tripleListTrain = openD(dirTrain)
    dirTest = "/home/e706/zhanxiangning/Tianchi/KnowledgeGraph/OpenBG500/OpenBG500_dev.tsv"
    tripleNumTest, tripleListTest = openD(dirTest)
    dirEntityVector = "/home/e706/zhanxiangning/KG/transE/entityVector.txt"
    entityVectorList, entityList = loadData(dirEntityVector)
    dirRelationVector = "/home/e706/zhanxiangning/KG/transE/relationVector.txt"
    relationVectorList, relationList = loadData(dirRelationVector)
    print("kaishitest")
  
    testHeadRaw = Test(entityList, entityVectorList, relationList, relationVectorList, tripleListTrain, tripleListTest)
    testHeadRaw.getRank()
    print(testHeadRaw.getMeanRank())
    testHeadRaw.writeRank("/home/e706/zhanxiangning/KG/transE/" + "testHeadRaw" + ".txt")
    testHeadRaw.getRelationRank()
    print(testHeadRaw.getMeanRank())
    testHeadRaw.writeRank("/home/e706/zhanxiangning/KG/transE/" + "testRelationRaw" + ".txt")

    testTailRaw = Test(entityList, entityVectorList, relationList, relationVectorList, tripleListTrain, tripleListTest, label = "tail")
    testTailRaw.getRank()
    print(testTailRaw.getMeanRank())
    testTailRaw.writeRank("/home/e706/zhanxiangning/KG/transE/" + "testTailRaw" + ".txt")

    testHeadFit = Test(entityList, entityVectorList, relationList, relationVectorList, tripleListTrain, tripleListTest, isFit = True)
    testHeadFit.getRank()
    print(testHeadFit.getMeanRank())
    testHeadFit.writeRank("/home/e706/zhanxiangning/KG/transE/" + "testHeadFit" + ".txt")
    testHeadFit.getRelationRank()
    print(testHeadFit.getMeanRank())
    testHeadFit.writeRank("/home/e706/zhanxiangning/KG/transE/" + "testRelationFit" + ".txt")

    testTailFit = Test(entityList, entityVectorList, relationList, relationVectorList, tripleListTrain, tripleListTest, isFit = True, label = "tail")
    testTailFit.getRank()
    print(testTailFit.getMeanRank())
    testTailFit.writeRank("/home/e706/zhanxiangning/KG/transE/" + "testTailFit" + ".txt")