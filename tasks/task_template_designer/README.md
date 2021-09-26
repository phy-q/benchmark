
## Run the Code

- To import the project (Note: Please use Unity version 2019.2.13f1 or later):
    - If in **Unity**: select **File->Open Project**  and select the root folder of science birds.
    - If in **Unity Hub**, if project is not added previously, click on **Add** button and select the root folder of science birds. Then click on the project which should have appeared in the list.
- To config build options:
    - Press **Shift+Ctrl+B** to open build settings. Select PC, Mac & Linux Standalone in Platform. 
    - Select the target platform in **Target Platform**
    - Click **Play Settings** and set Fullscreen Mode to **Windowed** 
    - Set Default Screen Width to **840** and Default Screen Height to **480**
- To run the code within Unity for debugging: 
    - From **Project** panel select **Scenes**->**GameWorld** and click the **play** button <img src="/Docs/PlayButton.png" height="15" /> in the middle of the top bar.
- To compile the code
    - Press **Ctrl+B** or select **File->Build and Run** to compile and run the code.
    - Building the game creates an executable. If no changes have been made to the game, you can later restart it by just running this file.
- To add/edit levels: Levels are found in `Assets/StreamingAssets/Levels`.
  The levels in the first directory seem to be compiled into the game when it is built; "streaming"
  levels can be edited/added without recompiling (though it currently requires restarting the game).
  These are found in the directory `test_Data/StreamingAssets/Levels` after the game has been built.


## Game Objects

The original Angry Birds has several game objects: birds, pigs, blocks, TNTs and other miscellaneous
objects used mainly for decoration. There are several types of birds and they vary in colour and size.
Some of them have special abilities that are triggered by touching the screen after a shot. The game also
has different types of pigs, each of which having different size and "health" points. Blocks have
different materials, what impact in their physical properties and "health points". TNTs are used to cause
explosions that deal damage in area, affecting several blocks and pigs. All these objects are placed on a
terrain that can be either completely flat or complex. Science Birds currently supports only part of these
objects:

- **Birds**
  - **Red**: Regular bird, no special abilities.
  - **Blue**: Splits into three birds when clicked, strong against ice blocks.
  - **Yellow**: Shoots forward at high speed when clicked, strong against wood blocks.
  - **Black**: Explodes when clicked or after impact, strong against stone blocks.
  - **White**: Drops explosive egg when clicked.
- **Pigs**:
  - **Small**
  - **Medium**
  - **Large**.
- **Blocks**:
  - **ice**
  - **stone**
  - **wood**.
- **TNT**
- **Terrain**

All these objects can be seen in the level shown in the figure above. It has three blocks of each material, three pigs, a TNT block and five birds (one of each type). Moreover, it has two rows of static square platforms floating in the air.

## Change Physical Parameters

Both global paramaters (e.g. gravity) and local paramaters (e.g. mass of a single type of objects) can be changed. These include but not limited to:
- Global
    - **gravity**: currently using default value so no code related to changing this parameter is presented
- Blocks
    - **mass**
        - Assets/Scripts/GameWorld/ABGameObject.cs
    - **_life**. A block will disappear if its _life <= 0 
        - Assets/Scripts/GameWorld/ABGameObject.cs
        - Assets/Scripts/GameWorld/ABBlock.cs
- Birds
    - **_woodDamage**, **_stoneDamage** and **_iceDamage**: damage ratio to different materials
        - Assets/Scripts/GameWorld/Characters/Birds/ABBird.cs (red bird uses default values)
        - Assets/Scripts/GameWorld/Characters/Birds/ABBirdBlue.cs
        - Assets/Scripts/GameWorld/Characters/Birds/ABBirdBlack.cs
        - Assets/Scripts/GameWorld/Characters/Birds/ABBirdWhite.cs
        - Assets/Scripts/GameWorld/Characters/Birds/ABBirdYellow.cs
    - **_launchGravity**: this parameter indicates the ratio that the object takes account the global gravity 
        - Locations same as above
    - **_launchForce**: this parameter indicates the impulse that the spring gives the birds while shooting it out 
        - Locations same as above
    - the initial **velocity** of the bird right after shooting is determined by both _launchGravity and _launchForce
    - Explosive parameters for black and white are described separately below.
- Explosive
    - **_explosionArea**, **_explosionPower** and **_explosionDamage**
        - Assets/Scripts/GameWorld/ABTNT.cs
        - Assets/Scripts/GameWorld/ABEgg.cs
        - Assets/Scripts/GameWorld/Characters/Birds/ABBirdBlack.cs
- Pigs 
    - **score** for killing a pig
        - Assets/Scripts/GameWorld/Characters/ABPig.cs
    - **mass** and **_life**
        - Assets/Scripts/GameWorld/ABGameObject.cs


## Level Representation

As mentioned before, levels are represented internally using a XML format. This format is basically
composed by a number of birds and a list of game objects, as shown in Figure 3. Each game object has
four attributes:

- **Type**: unique string representing the id of the object.
- **Material**: string defining the material of a block. Valid values are only "wood",
"stone" and "ice".
- **Position**: (x,y) float numbers representing the position of the game object. The origin (0,0) of the
coordinates system is the centre of the level.
- **Rotation**: float number that defines the rotation of the game object.

![Alt text](/Docs/Level1.png?raw=true "Level 1")

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

## Citing this Work

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
