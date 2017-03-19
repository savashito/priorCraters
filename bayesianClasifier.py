from priorHs import getSecondaryHsDistribution,getPrimaryHsDistribution
from priorShape import getSecondaryRmDistribution,getPrimaryRmDistribution
from priorD2 import getSecondaryDDistribution,getPrimaryDDistribution

print "hello mister goodman saul"

pPD = getPrimaryDDistribution()
pSD = getSecondaryDDistribution()
# print pSD.p(200)
# print pPD.p(200)
# print pSD.p(300)
# print pPD.p(300)
# print pSD.p(57)
# print pPD.p(57)

pSHs = getSecondaryHsDistribution()
pPHs = getPrimaryHsDistribution()
# print pSHs.p(0)
# print pPHs.p(0)
# print pSHs.p(0.1)

pRmP = getPrimaryRmDistribution()
pRmS = getSecondaryRmDistribution()
# print "Primary"
# print pRmP.p(1.05)
# print pRmP.p(0.8)
# print pRmP.p(1.2)
# print "secondary"
# print pRmS.p(1.05)
# print pRmS.p(0.8)
# print pRmS.p(1.2)