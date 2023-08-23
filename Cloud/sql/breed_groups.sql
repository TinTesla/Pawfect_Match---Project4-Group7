select da.Dog_species,da.temperament,da.popularity,da.max_height,da.energy_level_value,da.energy_level_category,
da.trainability_value,da.trainability_category,da.demeanor_value,da.demeanor_category from Dog_Info_AKC as da
where da.classlabel != '$breed_name' and da."group"
 in (
---get group of the selected breed
select akc."group" as grp_name from Dog_Info_AKC as akc
where akc.classlabel = '$breed_name'
)
