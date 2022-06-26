# Football match results prediction

Let's try to predict the outcomes of _Serie A_ football matches.

## Introduction

- The dataset contains *Serie A* matches starting from season 2005-06 to season 2020-21
- Cup matches (*Champions League*, *Europa League*, *Coppa Italia*) played over the course of each season were not taken
  into account

## Drawbacks

- We don't have data about new players that come to play in _Serie A_ during the course of the seasons. The model has to
  learn from zero context how important their contribution is for the outcome of the matches. If we were to considered
  multiple leagues, we could keep track of player transfers and maintain the history.
- We don't have data about cup matches played during the course of the seasons, like _Champions League_, _Europa League_
  and _Coppa Italia_. Since they are very prestigious competitions and matches are usually very competitive, teams put a
  lot of effort in them and therefore can then perform worse in the championship.
- We don't have any type of player performance metric like who scored a goal, who was the assist man, red or yellow
  cards, goalkeeper's saves etc. so the model could face some difficulties in learning which player is important for the
  team.