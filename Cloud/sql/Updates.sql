UPDATE Dog_Info_AKC 
SET popularity = (SELECT popularity
                  FROM DOG_INFO_AKC_WITH_IMAGES as img
                  WHERE img.dog_species = Dog_Info_AKC.dog_species);

UPDATE Dog_Info_AKC 
SET min_height = (SELECT min_height
                  FROM DOG_INFO_AKC_WITH_IMAGES as img
                  WHERE img.dog_species = Dog_Info_AKC.dog_species);
				  
UPDATE Dog_Info_AKC 
SET max_height= (SELECT max_height
                  FROM DOG_INFO_AKC_WITH_IMAGES as img
                  WHERE img.dog_species = Dog_Info_AKC.dog_species);
				  
UPDATE Dog_Info_AKC 
SET min_weight = (SELECT min_weight
                  FROM DOG_INFO_AKC_WITH_IMAGES as img
                  WHERE img.dog_species = Dog_Info_AKC.dog_species);


UPDATE Dog_Info_AKC 
SET max_weight = (SELECT max_weight
                  FROM DOG_INFO_AKC_WITH_IMAGES as img
                  WHERE img.dog_species = Dog_Info_AKC.dog_species);

UPDATE Dog_Info_AKC 
SET min_expectancy = (SELECT min_expectancy
                  FROM DOG_INFO_AKC_WITH_IMAGES as img
                  WHERE img.dog_species = Dog_Info_AKC.dog_species);


UPDATE Dog_Info_AKC 
SET max_expectancy = (SELECT max_expectancy
                  FROM DOG_INFO_AKC_WITH_IMAGES as img
                  WHERE img.dog_species = Dog_Info_AKC.dog_species);	
	
UPDATE Dog_Info_AKC 
SET popularity = (SELECT popularity
                  FROM DOG_INFO_AKC_WITH_IMAGES as img
                  WHERE img.dog_species = Dog_Info_AKC.dog_species);

UPDATE Dog_Info_AKC 
SET min_height = (SELECT min_height
                  FROM DOG_INFO_AKC_WITH_IMAGES as img
                  WHERE img.dog_species = Dog_Info_AKC.dog_species);
				  
UPDATE Dog_Info_AKC 
SET max_height= (SELECT max_height
                  FROM DOG_INFO_AKC_WITH_IMAGES as img
                  WHERE img.dog_species = Dog_Info_AKC.dog_species);
				  
UPDATE Dog_Info_AKC 
SET min_weight = (SELECT min_weight
                  FROM DOG_INFO_AKC_WITH_IMAGES as img
                  WHERE img.dog_species = Dog_Info_AKC.dog_species);


UPDATE Dog_Info_AKC 
SET max_weight = (SELECT max_weight
                  FROM DOG_INFO_AKC_WITH_IMAGES as img
                  WHERE img.dog_species = Dog_Info_AKC.dog_species);

UPDATE Dog_Info_AKC 
SET min_expectancy = (SELECT min_expectancy
                  FROM DOG_INFO_AKC_WITH_IMAGES as img
                  WHERE img.dog_species = Dog_Info_AKC.dog_species);


UPDATE Dog_Info_AKC 
SET Img_Link = (SELECT Img_Link
                  FROM DOG_INFO_AKC_WITH_IMAGES as img
                  WHERE img.dog_species = Dog_Info_AKC.dog_species);
	