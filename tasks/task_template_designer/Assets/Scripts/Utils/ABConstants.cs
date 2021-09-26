// SCIENCE BIRDS: A clone version of the Angry Birds game used for 
// research purposes
// 
// Copyright (C) 2016 - Lucas N. Ferreira - lucasnfe@gmail.com
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>
//

using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;

public enum MATERIALS
{
    wood,
    stone,
    ice,
    novelty

};

public enum BIRDS
{
    BirdRed,
    BirdBlue,
    BirdYellow,
    BirdBlack,
    BirdWhite
};

public enum PIGS
{
    BasicSmall,
    BasicMedium,
    BasicBig
};

public enum BLOCKS
{
    Circle,
    CircleSmall,
    RectBig,
    RectFat,
    RectMedium,
    RectSmall,
    RectTiny,
    SquareHole,
    SquareSmall,
    SquareTiny,
    Triangle,
    TriangleHole
};

public enum OBJECTS_SFX
{
    DAMAGE,
    DIE,
    FLYING,
    SELECTED,
    DRAGED,
    SHOT,
    MISC1,
    MISC2
}

public enum SLINGSHOT_LINE_POS
{
    SLING,
    BIRD
};

public class ABConstants
{

    public static readonly Vector3 SLING_SELECT_POS = new Vector3(0.15f, -0.8f, -1f);
    public static readonly Vector2 LEVEL_ORIGINAL_SIZE = new Vector2(17.5f, 11.58f);
    public static readonly int BLOCK_PARTCICLE_PER_SYSTEM = 25;
    public static readonly int BLOCK_PARTCICLE_SYSTEM_AMOUNT = 25;

    public static readonly string DEFAULT_LEVELS_FOLDER = "Levels";

    public const float LEVEL_IMMORTAL_TIME = 2f;

    public const float DAMAGE_CONSTANT_MULTIPLIER = 11.22f;

    public const float BIRD_MAX_LANUCH_SPEED = 10f;

    public const float DENSITY_WOOD = 1.5f;
    public const float DENSITY_STONE = 4f;
    public const float DENSITY_ICE = 0.75f;

    public const float DEFENSE_WOOD = 2.5f;
    public const float DEFENSE_STONE = 15f;
    public const float DEFENSE_ICE = 2f;

#if UNITY_STANDALONE_OSX && !UNITY_EDITOR

	public static readonly float MOUSE_SENSIBILITY = 5f;
	public static readonly string STREAMINGASSETS_FOLDER = Application.dataPath + "/Resources/Data/StreamingAssets"; 
	public static readonly string CUSTOM_LEVELS_FOLDER = "/Resources/Data/StreamingAssets/Levels";
	public static readonly string DEFAULT_ASSET_BUNDLE_FILE = Application.dataPath + "/Resources/Data/StreamingAssets/AssetBundle/assetbundle";

	public static readonly string colorDataFile = Application.dataPath + "/Resources/Data/StreamingAssets/ColorCodes/BaseMap.json";
    public static readonly string assetBundleOSPath = "/AssetBundle/OSX/assetbundle";

#elif UNITY_STANDALONE_WIN && !UNITY_EDITOR

	public static readonly float MOUSE_SENSIBILITY = 25f;
	public static readonly string STREAMINGASSETS_FOLDER =Application.dataPath + "/StreamingAssets"; 
	public static readonly string CUSTOM_LEVELS_FOLDER = "/StreamingAssets/Levels";
	public static readonly string colorDataFile = Application.dataPath + "/StreamingAssets/ColorCodes/BaseMap.json";
	public static readonly string DEFAULT_ASSET_BUNDLE_FILE = Application.dataPath + "/StreamingAssets/AssetBundle/assetbundle";
	public static readonly string assetBundleOSPath = "/AssetBundle/windows/assetbundle";
#else

    public static readonly float MOUSE_SENSIBILITY = 0.65f;
    public static readonly string STREAMINGASSETS_FOLDER = Application.dataPath + "/StreamingAssets";

    public static readonly string CUSTOM_LEVELS_FOLDER = "/StreamingAssets/Levels";
    public static readonly string DEFAULT_ASSET_BUNDLE_FILE = Application.dataPath + "/StreamingAssets/AssetBundle/assetbundle";
    public static readonly string colorDataFile = Application.dataPath + "/StreamingAssets/ColorCodes/BaseMap.json";
    public static readonly string assetBundleOSPath = "/AssetBundle/linux/assetbundle";

#endif
}

public class ABWorldAssets
{

    public static readonly GameObject[] WOOD_DESTRUCTION_EFFECT = Resources.LoadAll<GameObject>("Prefabs/GameWorld/Particles/Wood");
    public static readonly GameObject[] STONE_DESTRUCTION_EFFECT = Resources.LoadAll<GameObject>("Prefabs/GameWorld/Particles/Stone");
    public static readonly GameObject[] ICE_DESTRUCTION_EFFECT = Resources.LoadAll<GameObject>("Prefabs/GameWorld/Particles/Stone");
    public static readonly GameObject[] TRAIL_PARTICLES = Resources.LoadAll<GameObject>("Prefabs/GameWorld/Particles/BirdTrail");

    public static readonly Dictionary<string, GameObject> BIRDS = LevelLoader.LoadABResource("Prefabs/GameWorld/Characters/Birds");
    public static readonly Dictionary<string, GameObject> PIGS = LevelLoader.LoadABResource("Prefabs/GameWorld/Characters/Pigs");
    public static readonly Dictionary<string, GameObject> BLOCKS = LevelLoader.LoadABResource("Prefabs/GameWorld/Blocks");


    public static readonly GameObject GROUND_EXTENSION = (GameObject)Resources.Load("Prefabs/GameWorld/GroundExtension");
    public static readonly GameObject LANDSCAPE = (GameObject)Resources.Load("Prefabs/GameWorld/Landscape");
    public static readonly GameObject BACKGROUND = (GameObject)Resources.Load("Prefabs/GameWorld/Background");
    public static readonly GameObject SLINGSHOT = (GameObject)Resources.Load("Prefabs/GameWorld/Slingshot");
    public static readonly GameObject TNT = (GameObject)Resources.Load("Prefabs/GameWorld/TNT");
    public static readonly GameObject EGG = (GameObject)Resources.Load("Prefabs/GameWorld/Particles/Egg");
    public static readonly GameObject PLATFORM = (GameObject)Resources.Load("Prefabs/GameWorld/Platform");
    public static readonly GameObject SCORE_POINT = (GameObject)Resources.Load("Prefabs/GameWorld/ScorePoints");

    public static readonly PhysicsMaterial2D WOOD_MATERIAL = Resources.Load("Materials/Wood") as PhysicsMaterial2D;
    public static readonly PhysicsMaterial2D STONE_MATERIAL = Resources.Load("Materials/Stone") as PhysicsMaterial2D;
    public static readonly PhysicsMaterial2D ICE_MATERIAL = Resources.Load("Materials/Ice") as PhysicsMaterial2D;
}
