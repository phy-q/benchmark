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
using UnityEditor;
using System;
using System.Reflection;
using System.Collections;
using System.Collections.Generic;
using UnityEditor.SceneManagement;

class LevelEditor : EditorWindow
{

    public static BIRDS _birdsOps;
    public static PIGS _pigsOps;
    public static BLOCKS _blocksOps;
    public static MATERIALS _material;

    private static Dictionary<string, GameObject> _birds;
    private static Dictionary<string, GameObject> _pigs;
    private static Dictionary<string, GameObject> _blocks;
    private static GameObject _platform;
    private static GameObject _tnt;

    private static int _birdsAdded = 0;
    private static int _birdsAmounInARow = 5;
    private static Vector3 _groundPos;

    [MenuItem("Level Editor/Edit Level")]
    static void Init()
    {

        if (EditorSceneManager.GetActiveScene().name != "GameWorld")
        {

            if (EditorUtility.DisplayDialog("Warning", "This window can only be opened in the Game World scene. " +
               "Do you wanna go to that scene?", "Yes", "Cancel"))
            {

                EditorSceneManager.OpenScene("Assets/Scenes/GameWorld.unity");
            }
            else
                return;
        }

        LevelEditor window = (LevelEditor)EditorWindow.GetWindow(typeof(LevelEditor));
        window.Show();

    }

    void OnEnable()
    {
        _birds = LevelLoader.LoadABResource("Prefabs/GameWorld/Characters/Birds");
        _pigs = LevelLoader.LoadABResource("Prefabs/GameWorld/Characters/Pigs");
        _blocks = LevelLoader.LoadABResource("Prefabs/GameWorld/Blocks");
        _platform = Resources.Load("Prefabs/GameWorld/Platform") as GameObject;
        _tnt = Resources.Load("Prefabs/GameWorld/TNT") as GameObject;

        //_groundPos = new Vector3 (0f, -2.74f, 0f);
        _groundPos = new Vector3(0f, -3.26f, 0f);

        _birdsAdded = GameObject.Find("Birds").transform.childCount;
    }

    void OnFocus()
    {

        // ToggleGizmos(false); commented to avoid the error when opening the level editor window

        HideLayer(LayerMask.NameToLayer("UI"));
    }

    void OnHierarchyChange()
    {

        if (EditorSceneManager.GetActiveScene().name != "GameWorld")
            Close();
    }

    void OnGUI()
    {
        EditorGUILayout.BeginHorizontal();

        _birdsOps = (BIRDS)EditorGUILayout.EnumPopup("", _birdsOps);
        if (GUILayout.Button("Create Bird", GUILayout.Width(80), GUILayout.Height(20)))
        {

            CreateBird(_birdsOps.ToString());
        }

        EditorGUILayout.EndVertical();

        // Pigs section
        EditorGUILayout.BeginHorizontal();

        _pigsOps = (PIGS)EditorGUILayout.EnumPopup("", _pigsOps);

        if (GUILayout.Button("Create Pig", GUILayout.Width(80), GUILayout.Height(20)))
        {

            GameObject pig = InstantiateGameObject(_pigs[_pigsOps.ToString()]);
            pig.transform.parent = GameObject.Find("Blocks").transform;
        }

        EditorGUILayout.EndVertical();

        // Blocks section
        EditorGUILayout.BeginHorizontal();

        _blocksOps = (BLOCKS)EditorGUILayout.EnumPopup("", _blocksOps);
        _material = (MATERIALS)EditorGUILayout.EnumPopup("", _material);

        if (GUILayout.Button("Create Block", GUILayout.Width(80), GUILayout.Height(20)))
        {

            GameObject block = InstantiateGameObject(_blocks[_blocksOps.ToString()]);
            block.transform.parent = GameObject.Find("Blocks").transform;

            BlockEditor.UpdateBlockMaterial(block.GetComponent<ABBlock>(), _material);
        }

        EditorGUILayout.EndHorizontal();

        if (GUILayout.Button("Create Platform"))
        {

            GameObject platform = InstantiateGameObject(_platform);
            platform.transform.parent = GameObject.Find("Platforms").transform;
        }

        if (GUILayout.Button("Create TNT"))
        {

            GameObject tnt = InstantiateGameObject(_tnt);
            tnt.transform.parent = GameObject.Find("Blocks").transform;
        }

        EditorGUILayout.BeginHorizontal();

        if (GUILayout.Button("Load Level"))
            LoadLevel();

        if (GUILayout.Button("New Level"))
        {

            if (EditorUtility.DisplayDialog("Create New Level",
                   "Are you sure you want to create a new level? This operation will " +
                "not save the current level in the scene", "Yes", "Cancel"))
            {

                CleanLevel();
            }
        }

        if (GUILayout.Button("Save Level"))
            SaveLevel();

        EditorGUILayout.EndHorizontal();
    }

    GameObject InstantiateGameObject(GameObject[] source, int index)
    {

        GameObject cube = (GameObject)PrefabUtility.InstantiatePrefab(source[index]);
        cube.transform.position = Vector3.zero;

        return cube;
    }

    GameObject InstantiateGameObject(GameObject source)
    {

        GameObject cube = (GameObject)PrefabUtility.InstantiatePrefab(source);
        // cube.transform.position = Vector3.zero;
        cube.transform.position = new Vector3(0.0f, -3.0f, 0.0f);

        return cube;
    }

    void CleanLevel()
    {

        _birdsAdded = 0;

        DestroyImmediate(GameObject.Find("Birds").gameObject);
        GameObject birds = new GameObject();
        birds.name = "Birds";
        birds.transform.parent = GameObject.Find("GameWorld").transform;

        DestroyImmediate(GameObject.Find("Blocks").gameObject);
        GameObject blocks = new GameObject();
        blocks.name = "Blocks";
        blocks.transform.parent = GameObject.Find("GameWorld").transform;

        DestroyImmediate(GameObject.Find("Platforms").gameObject);
        GameObject plats = new GameObject();
        plats.name = "Platforms";
        plats.transform.parent = GameObject.Find("GameWorld").transform;
    }

    void LoadLevel()
    {

        string path = EditorUtility.OpenFilePanel("Select level to open", "Assets/Resources/Levels/", "");

        if (path != "")
        {

            CleanLevel();

            Debug.Log(path);

            // TextAsset levelFile = (TextAsset)Resources.Load (finalPath);
            string levelText = LevelLoader.ReadXmlLevel(path);
            // Debug.Log(levelText);
            ABLevel level = LevelLoader.LoadXmlLevel(levelText);

            DecodeLevel(level);
        }
    }

    void SaveLevel()
    {

        string path = EditorUtility.SaveFilePanel("Select location to save level", "Assets/Resources/Levels/", "level.xml", "xml");

        if (path != "")
        {

            LevelLoader.SaveXmlLevel(EncodeLevel(), path);
        }
    }

    void CreateBird(string type)
    {

        GameObject bird = InstantiateGameObject(_birds[type]);
        bird.name = bird.name + "_" + _birdsAdded;
        bird.transform.parent = GameObject.Find("Birds").transform;

        //Vector3 birdsPos = ABConstants.SLING_SELECT_POS;
        Vector3 firstBirdsPos = new Vector3(-8.12f, -1.7f, 1f);
        Vector3 birdsPos = firstBirdsPos;

        // From the second Bird on, they are added to the ground
        if (_birdsAdded >= 1)
        {
            birdsPos.y = _groundPos.y;

            for (int i = 0; i < _birdsAdded; i++)
            {

                if ((i + 1) % _birdsAmounInARow == 0)
                {

                    float coin = (UnityEngine.Random.value < 0.5f ? 1f : -1);
                    birdsPos.x = firstBirdsPos.x + (UnityEngine.Random.value * 0.5f * coin);
                }

                birdsPos.x -= bird.GetComponent<SpriteRenderer>().bounds.size.x * 1.75f;
            }
        }

        bird.transform.position = birdsPos;

        _birdsAdded++;
    }

    /*
	public ABLevel EncodeLevel() 
	{
		ABLevel level = new ABLevel();

		level.birds = new List<BirdData>(); 
		level.pigs = new List<OBjData>();
		level.blocks = new List<BlockData>();
		level.platforms = new List<PlatData>();

		foreach (Transform child in GameObject.Find ("Birds").transform) {

			string type = child.name;
			level.birds.Add (new BirdData (type));
		}

		foreach (Transform child in GameObject.Find ("Blocks").transform) {

			string type = child.name;
			float x = child.transform.position.x;
			float y = child.transform.position.y;
			float rotation = child.transform.rotation.eulerAngles.z;

			if (child.GetComponent<ABPig> () != null) {

				level.pigs.Add (new OBjData (type, rotation, x, y));
			} 
			else if (child.GetComponent<ABBlock> () != null) {
				
				string material = child.GetComponent<ABBlock> ()._material.ToString ();
				level.blocks.Add (new BlockData (type, rotation, x, y, material));
			}

		}

		foreach (Transform child in GameObject.Find ("Platforms").transform) {

			PlatData obj = new PlatData ();

			obj.type = child.name;
			obj.x = child.transform.position.x;
			obj.y = child.transform.position.y;
			obj.rotation = child.transform.rotation.eulerAngles.z;
			obj.scaleX = child.transform.localScale.x;
			obj.scaleY = child.transform.localScale.y;

			level.platforms.Add (obj);
		}
			
		return level;
	}
    */

    public static ABLevel EncodeLevel()
    {
        Debug.Log("Encoding the level");

        ABLevel level = new ABLevel();

        level.birds = new List<BirdData>();
        level.pigs = new List<OBjData>();
        level.blocks = new List<BlockData>();
        level.platforms = new List<PlatData>();

        foreach (Transform child in GameObject.Find("Birds").transform)
        {

            string type = child.name;
            string spriteName = child.GetComponent<SpriteRenderer>().sprite.name;

            // skip blue bird's children and get the correct bird type from the sprite name
            if (type.Contains("BlueBird_Child"))
            {
                continue;
            }
            else if (spriteName.Contains("bird_blue"))
            {
                type = "BirdBlue";
            }
            else if (spriteName.Contains("bird_red"))
            {
                type = "BirdRed";
            }
            else if (spriteName.Contains("bird_yellow"))
            {
                type = "BirdYellow";
            }
            else if (spriteName.Contains("bird_black"))
            {
                type = "BirdBlack";
            }
            else if (spriteName.Contains("bird_white"))
            {
                type = "BirdWhite";
            }
            else
            {
                continue;
            }

            level.birds.Add(new BirdData(type));
        }

        foreach (Transform child in GameObject.Find("Blocks").transform)
        {

            string type = child.name;
            float x = child.transform.position.x;
            float y = child.transform.position.y;
            float rotation = child.transform.rotation.eulerAngles.z;

            if (child.GetComponent<ABPig>() != null)
            {

                level.pigs.Add(new OBjData(type, rotation, x, y));
            }
            else if (child.GetComponent<ABBlock>() != null)
            {

                string material = child.GetComponent<ABBlock>()._material.ToString();
                level.blocks.Add(new BlockData(type, rotation, x, y, material));
            }
            else if (child.GetComponent<ABTNT>() != null)
            {
                level.tnts.Add(new OBjData("TNT", rotation, x, y));
            }

        }

        foreach (Transform child in GameObject.Find("Platforms").transform)
        {

            PlatData obj = new PlatData();

            obj.type = child.name;
            obj.x = child.transform.position.x;
            obj.y = child.transform.position.y;
            obj.rotation = child.transform.rotation.eulerAngles.z;
            obj.scaleX = child.transform.localScale.x;
            obj.scaleY = child.transform.localScale.y;

            level.platforms.Add(obj);
        }

        return level;
    }

    public void DecodeLevel(ABLevel level)
    {
        foreach (OBjData gameObj in level.pigs)
        {

            Vector2 pos = new Vector2(gameObj.x, gameObj.y);
            Quaternion rotation = Quaternion.Euler(0, 0, gameObj.rotation);

            GameObject pig = InstantiateGameObject(_pigs[gameObj.type]);
            pig.transform.parent = GameObject.Find("Blocks").transform;
            pig.transform.position = pos;
            pig.transform.rotation = rotation;
        }

        _birdsAdded = 0;
        foreach (BirdData gameObj in level.birds)
        {

            CreateBird(gameObj.type);
            _birdsAdded++;
        }

        foreach (BlockData gameObj in level.blocks)
        {

            Vector2 pos = new Vector2(gameObj.x, gameObj.y);
            Quaternion rotation = Quaternion.Euler(0, 0, gameObj.rotation);

            GameObject block = InstantiateGameObject(_blocks[gameObj.type]);
            block.transform.parent = GameObject.Find("Blocks").transform;
            block.transform.position = pos;
            block.transform.rotation = rotation;

            MATERIALS material = (MATERIALS)Enum.Parse(typeof(MATERIALS), gameObj.material);
            BlockEditor.UpdateBlockMaterial(block.GetComponent<ABBlock>(), material);
        }

        foreach (PlatData gameObj in level.platforms)
        {
            Vector2 pos = new Vector2(gameObj.x, gameObj.y);
            Quaternion rotation = Quaternion.Euler(0, 0, gameObj.rotation);
            Vector3 scale = new Vector3(gameObj.scaleX, gameObj.scaleY, 0);

            GameObject platform = InstantiateGameObject(_platform);
            platform.transform.parent = GameObject.Find("Platforms").transform;
            platform.transform.position = pos;
            platform.transform.rotation = rotation;
            platform.transform.localScale = scale;

        }

        foreach (OBjData gameObj in level.tnts)
        {
            Debug.Log("TNT in level");
            Vector2 pos = new Vector2(gameObj.x, gameObj.y);
            Quaternion rotation = Quaternion.Euler(0, 0, gameObj.rotation);

            GameObject tnt = InstantiateGameObject(_tnt);
            tnt.transform.parent = GameObject.Find("Blocks").transform;
            tnt.transform.position = pos;
            tnt.transform.rotation = rotation;

        }
    }

    public static void ToggleGizmos(bool gizmosOn)
    {

        int val = gizmosOn ? 1 : 0;

        Assembly asm = Assembly.GetAssembly(typeof(Editor));
        Type type = asm.GetType("UnityEditor.AnnotationUtility");

        if (type != null)
        {

            MethodInfo getAnnotations = type.GetMethod("GetAnnotations", BindingFlags.Static | BindingFlags.NonPublic);
            MethodInfo setGizmoEnabled = type.GetMethod("SetGizmoEnabled", BindingFlags.Static | BindingFlags.NonPublic);
            MethodInfo setIconEnabled = type.GetMethod("SetIconEnabled", BindingFlags.Static | BindingFlags.NonPublic);

            var annotations = getAnnotations.Invoke(null, null);

            foreach (object annotation in (IEnumerable)annotations)
            {

                Type annotationType = annotation.GetType();
                FieldInfo classIdField = annotationType.GetField("classID", BindingFlags.Public | BindingFlags.Instance);
                FieldInfo scriptClassField = annotationType.GetField("scriptClass", BindingFlags.Public | BindingFlags.Instance);

                if (classIdField != null && scriptClassField != null)
                {
                    int classId = (int)classIdField.GetValue(annotation);

                    string scriptClass = (string)scriptClassField.GetValue(annotation);
                    setGizmoEnabled.Invoke(null, new object[] { classId, scriptClass, val });
                    setIconEnabled.Invoke(null, new object[] { classId, scriptClass, val });
                }
            }
        }
    }

    static void HideLayer(int layerNumber)
    {
        LayerMask layerNumberBinary = 1 << layerNumber;
        LayerMask blocksLayer = 1 << LayerMask.NameToLayer("Blocks");
        LayerMask birdsLayer = 1 << LayerMask.NameToLayer("Birds");
        LayerMask pigsLayer = 1 << LayerMask.NameToLayer("Pigs");
        LayerMask platLayer = 1 << LayerMask.NameToLayer("Platforms");

        Tools.visibleLayers = ~layerNumberBinary;
        Tools.lockedLayers = ~(blocksLayer | birdsLayer | pigsLayer | platLayer);

        SceneView.RepaintAll();
    }


}
