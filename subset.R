# Pre-Processing
# Goal: to subset the data to only have reviews that have star ratings more than one standard deviation away from the mean star rating the user gives


library(tibble)
library(dplyr)

Data_Final <- read.csv("Data_Final.csv")


summarized <- Data_Final %>% 
              group_by(User_id) %>% 
              summarise(meanS = mean(Star), sdS = sd(Star)) %>% 
              arrange(desc(User_id)) 

#Summarized becomes each user, their average star rating they give, and their standard deviation rating they give


deviations <- Data_Final %>% 
              left_join(summarized, by = "User_id")  %>% 
              filter(Star > (meanS + sdS) | Star < (meanS - sdS))

#Joining back with the dataset, this subsets the reviews to only the Stars that are more than one standard deviation away from mean

# Observations goes from 53,845 to 11,821

write.csv(deviations, file = 'data_subset.csv')
