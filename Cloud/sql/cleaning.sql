select DISTINCT dgi.dog_species,dgi.classlabel
from DOG_INFO as dgi
where dgi.classlabel in
(select dg.classlabel
from DOG_INFO as dg
group by dg.classlabel
having count(distinct dg.dog_species) > 1) 
order by dgi.classlabel