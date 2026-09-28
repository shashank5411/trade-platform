import json,sys
d=json.load(sys.stdin)
t=d['Table']
print('Location:', t['StorageDescriptor']['Location'])
print('Columns:')
for c in t['StorageDescriptor']['Columns']:
    print(' ', c['Name'], '-', c['Type'])
print('Partition keys:')
for p in t.get('PartitionKeys',[]):
    print(' ', p['Name'], '-', p['Type'])
