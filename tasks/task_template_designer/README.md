## Task Template Designer

This work is a modification of a cover version of the "Angry Birds" game used for research purposes developed by Lucas Ferreira. If you use this clone in your research, please cite:

```
@inproceedings{ferreira_2014_a,
    author = {Lucas Ferreira and Claudio Toledo},
    title = {A Search-based Approach for Generating Angry Birds Levels},
    booktitle = {Proceedings of the 9th IEEE International Conference on Computational Intelligence in Games},
    series = {CIG'14},
    year = {2014},
    location = {Dortmund, Germany}
}
```

## Run the Code

- To import the project (Note: Please use Unity version 2019.2.13f1 or later):
    - If in **Unity**: select **File->Open Project**  and select the root folder of science birds.
    - If in **Unity Hub**, if project is not added previously, click on **Add** button and select the root folder of science birds. Then click on the project which should have appeared in the list.
- To open the Task Template Designer and create your own task template:
    - Run the application in Unity Editor and load any game level. 
    - While in the game level, open the Level Editor menu by navigating to the Level Editor -> Edit Level in the top-menu of the Unity editor.
    - From the Level Editor menu you can load a game level, save the level, and add any game objects to the level.
    - Design the template by adding new game objects, adjusting their positions, and resizing them as you wish.
    - After designing the task template, save the template using the Save Level button in the Level Editor menu.


## Task/Task Template Representation

Tasks are represented internally using a XML format. This format is basically composed by a number of birds and a list of game objects. Each game object has four attributes:

- **Type**: unique string representing the id of the object.
- **Material**: string defining the material of a block. Valid values are only "wood",
"stone" and "ice".
- **Position**: (x,y) float numbers representing the position of the game object. The origin (0,0) of the
coordinates system is the centre of the level.
- **Rotation**: float number that defines the rotation of the game object.

```
<?xml version="1.0" encoding="utf-16"?>
<Level>
  <Camera x="0" y="-1" minWidth="15" maxWidth="17.5">
  <Birds>
  	<Bird type="BirdRed"/>
  	<Bird type="BirdRed"/>
  	<Bird type="BirdBlue"/>
  </Birds>
  <Slingshot x="-5" y="-2.5">
  <GameObjects>
    <Block type="RectMedium" material="wood" x="5.25" y="-3.23" rotation="0" />
    <Block type="RectFat" material="wood" x="5.22" y="-2.71" rotation="90.00001" />
    <Pig type="BasicMedium" x="5.21" y="-2.03" rotation="0" />
    <TNT type="" x="3.21" y="-4" rotation="0" />
  </GameObjects>
</Level>
```

