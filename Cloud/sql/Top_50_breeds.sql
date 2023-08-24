select cast(src.popularity as INT) as score,* from 
(select * from DOG_INFO where popularity is not NULL) as src
where cast(src.popularity as INT) between 1 and 50
order by cast(src.popularity as INT) ASC

