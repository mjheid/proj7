### Proj7

```sh
cat data/data.ped | cut -f 1 | sort | uniq > data/countries.txt
```

```sh
cat data/MARITIME_ROUTE.ped | sed 's/Sahara_OCC/Sahara/g' | sed 's/East_Rumelia/Rumelia/g'  > data/data.ped
```