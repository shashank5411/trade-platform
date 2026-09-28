import sys
sys.path.insert(0, ".")
from query.athena import query
from query import api

db = api.DB["sec"]
sql1 = "SELECT entity, COUNT(*) as rows FROM documents GROUP BY entity ORDER BY entity"
print(query(sql1, db))
