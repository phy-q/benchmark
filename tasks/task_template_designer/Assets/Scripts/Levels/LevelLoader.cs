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

﻿using UnityEngine;
using System;
using System.IO;
using System.Xml;
using System.Text;
using System.Collections.Generic;

public class LevelLoader {
	
	public static string ReadXmlLevel(string path) {
	
		string xmlText = "";

		if (path.Contains ("StreamingAssets")) {

			xmlText = File.ReadAllText (path);
		} 
		else {

			string[] stringSeparators = new string[] {"Levels/"};
			string[] arrayPath = path.Split (stringSeparators, StringSplitOptions.None);
			string finalPath = arrayPath [1].Split ('.') [0];

			TextAsset levelData = Resources.Load<TextAsset>("Levels/" + finalPath);
			xmlText = levelData.text;
		}

		return xmlText;
	}
	
	public static ABLevel LoadXmlLevel(string xmlString) {

		ABLevel level = new ABLevel();

		using (XmlReader reader = XmlReader.Create(new StringReader(xmlString)))
		{
			
			reader.ReadToFollowing("Level");

			level.width = 1;
			if (reader.GetAttribute ("width") != null) {

				reader.MoveToAttribute("width");
				level.width = (int)Convert.ToInt32 (reader.Value);
			}

			reader.ReadToFollowing("Camera");

			reader.MoveToAttribute("x");
			level.camera.x = (float)Convert.ToDouble (reader.Value);

			reader.MoveToAttribute("y");
			level.camera.y = (float)Convert.ToDouble (reader.Value);

			reader.MoveToAttribute("minWidth");
			level.camera.minWidth = (float)Convert.ToDouble (reader.Value);

			reader.MoveToAttribute("maxWidth");
			level.camera.maxWidth = (float)Convert.ToDouble (reader.Value);
			
            //Reading the score
            reader.ReadToFollowing("Score");
            reader.MoveToAttribute("highScore");
            level.score.highScore = (float)Convert.ToDouble(reader.Value);

            reader.ReadToFollowing("Birds");
			reader.Read ();

			while (reader.Read ()) {

				string nodeName = reader.LocalName;
				if (nodeName == "Birds")
					break;

				reader.MoveToAttribute("type");
				string type = reader.Value;

				level.birds.Add (new BirdData (type));
				reader.Read ();
			}

			reader.ReadToFollowing("Slingshot");

			reader.MoveToAttribute("x");
			level.slingshot.x = (float)Convert.ToDouble (reader.Value);

			reader.MoveToAttribute("y");
			level.slingshot.y = (float)Convert.ToDouble (reader.Value);
            /* ### commented by Chathura to avoid issues with loading levels from level editor ### what does this code do?
			if (reader.ReadToFollowing("assetBundle")){
				if(reader.GetAttribute("path") != null){
					reader.MoveToAttribute("path");
					level.assetBundleFilePath = reader.Value; 
				}
			}
			*/
            reader.ReadToFollowing("GameObjects");
			reader.Read ();

			while (reader.Read())
			{
				string nodeName = reader.LocalName;
				if (nodeName == "GameObjects")
					break;

				reader.MoveToAttribute("type");
				string type = reader.Value;

				string material = "";
				if (reader.GetAttribute ("material") != null) {

					reader.MoveToAttribute("material");
					material = reader.Value;
				}
					
				reader.MoveToAttribute("x");
				float x = (float)Convert.ToDouble(reader.Value);

				reader.MoveToAttribute("y");
				float y = (float)Convert.ToDouble(reader.Value);

				float rotation = 0f;
				if (reader.GetAttribute ("rotation") != null) {
				
					reader.MoveToAttribute ("rotation");
					rotation = (float)Convert.ToDouble (reader.Value);
				}

				if (nodeName == "Block") {

					level.blocks.Add (new BlockData (type, rotation, x, y, material));
					reader.Read ();
				} 
				else if (nodeName == "Pig") {

					level.pigs.Add (new OBjData (type, rotation, x, y));
					reader.Read ();
				}
				else if (nodeName == "TNT") {

					level.tnts.Add (new OBjData (type, rotation, x, y));
					reader.Read ();
				}
				else if (nodeName == "Platform") {

					float scaleX = 1f;
					if (reader.GetAttribute ("scaleX") != null) {

						reader.MoveToAttribute ("scaleX");
						scaleX = (float)Convert.ToDouble (reader.Value);
					}

					float scaleY = 1f;
					if (reader.GetAttribute ("scaleY") != null) {

						reader.MoveToAttribute ("scaleY");
						scaleY = (float)Convert.ToDouble (reader.Value);
					}

					level.platforms.Add (new PlatData (type, rotation, x, y, scaleX, scaleY));
					reader.Read ();
				}
			}

		}

		return level;
	}

    /*
	public static void SaveXmlLevel(ABLevel level, string path) {

		StringBuilder output = new StringBuilder();
		XmlWriterSettings ws = new XmlWriterSettings();
		ws.Indent = true;

		using (XmlWriter writer = XmlWriter.Create(output, ws))
		{
			writer.WriteStartElement("Level");
			writer.WriteAttributeString("width", level.width.ToString());

			writer.WriteStartElement("Camera");
			writer.WriteAttributeString("x", level.camera.x.ToString());
			writer.WriteAttributeString("y", level.camera.y.ToString());
			writer.WriteAttributeString("minWidth", level.camera.minWidth.ToString());
			writer.WriteAttributeString("maxWidth", level.camera.maxWidth.ToString());
			writer.WriteEndElement();

            //Writing score
            writer.WriteStartElement("Score");
            writer.WriteAttributeString("highScore", level.score.highScore.ToString());
            writer.WriteEndElement();

            writer.WriteStartElement("Birds");

			foreach(BirdData abBird in level.birds)
			{
				writer.WriteStartElement("Bird");
				writer.WriteAttributeString("type", abBird.type.ToString());
				writer.WriteEndElement();
			}

			writer.WriteStartElement("Slingshot");
			writer.WriteAttributeString("x", level.slingshot.x.ToString());
			writer.WriteAttributeString("y", level.slingshot.y.ToString());
			writer.WriteEndElement();

            

            writer.WriteStartElement("GameObjects");

			foreach(BlockData abObj in level.blocks)
			{
				writer.WriteStartElement("Block");
				writer.WriteAttributeString("type", abObj.type.ToString());
				writer.WriteAttributeString("material", abObj.material.ToString());
				writer.WriteAttributeString("x", abObj.x.ToString());
				writer.WriteAttributeString("y", abObj.y.ToString());
				writer.WriteAttributeString("rotation", abObj.rotation.ToString());
				writer.WriteEndElement();
			}

			foreach(OBjData abObj in level.pigs)
			{
				writer.WriteStartElement("Pig");
				writer.WriteAttributeString("type", abObj.type.ToString());
				writer.WriteAttributeString("x", abObj.x.ToString());
				writer.WriteAttributeString("y", abObj.y.ToString());
				writer.WriteAttributeString("rotation", abObj.rotation.ToString());
				writer.WriteEndElement();
			}

			foreach(OBjData abObj in level.tnts)
			{
				writer.WriteStartElement("TNT");
				writer.WriteAttributeString("type", abObj.type.ToString());
				writer.WriteAttributeString("x", abObj.x.ToString());
				writer.WriteAttributeString("y", abObj.y.ToString());
				writer.WriteAttributeString("rotation", abObj.rotation.ToString());
				writer.WriteEndElement();
			}

			foreach(PlatData abObj in level.platforms)
			{
				writer.WriteStartElement("Platform");
				writer.WriteAttributeString("type", abObj.type.ToString());
				writer.WriteAttributeString("x", abObj.x.ToString());
				writer.WriteAttributeString("y", abObj.y.ToString());
				writer.WriteAttributeString("rotation", abObj.rotation.ToString());
				writer.WriteAttributeString("scaleX", abObj.scaleX.ToString());
				writer.WriteAttributeString("scaleY", abObj.scaleY.ToString());
				writer.WriteEndElement();
			}
		}
			
		StreamWriter streamWriter = new StreamWriter(path);
		streamWriter.WriteLine(output.ToString());
		streamWriter.Close();
	}
		*/


    public static void SaveXmlLevel(ABLevel level, string path)
    {

        Debug.Log("Writing the level file");

        // number of decimal places needed to be written
        int precisionPoints = 5;

        StringBuilder output = new StringBuilder();
        XmlWriterSettings ws = new XmlWriterSettings();
        ws.Indent = true;

        using (XmlWriter writer = XmlWriter.Create(output, ws))
        {
            writer.WriteStartElement("Level");
            //writer.WriteAttributeString("width", level.width.ToString());
            writer.WriteAttributeString("width", "2");

            writer.WriteStartElement("Camera");
            //writer.WriteAttributeString("x", level.camera.x.ToString());
            //writer.WriteAttributeString("y", level.camera.y.ToString());
            //writer.WriteAttributeString("minWidth", level.camera.minWidth.ToString());
            //writer.WriteAttributeString("maxWidth", level.camera.maxWidth.ToString());
            writer.WriteAttributeString("x", "0");
            writer.WriteAttributeString("y", "-1");
            writer.WriteAttributeString("minWidth", "25");
            writer.WriteAttributeString("maxWidth", "35");
            writer.WriteEndElement();

            //Writing score
            writer.WriteStartElement("Score");
            writer.WriteAttributeString("highScore", level.score.highScore.ToString());
            writer.WriteEndElement();

            writer.WriteStartElement("Birds");
            foreach (BirdData abBird in level.birds)
            {

                writer.WriteStartElement("Bird");
                writer.WriteAttributeString("type", abBird.type.ToString());
                writer.WriteEndElement();
            }
            writer.WriteEndElement();

            writer.WriteStartElement("Slingshot");
            //writer.WriteAttributeString("x", level.slingshot.x.ToString());
            //writer.WriteAttributeString("y", level.slingshot.y.ToString());
            writer.WriteAttributeString("x", "-12.0");
            writer.WriteAttributeString("y", "-2.5");
            writer.WriteEndElement();



            writer.WriteStartElement("GameObjects");

            foreach (BlockData abObj in level.blocks)
            {
                writer.WriteStartElement("Block");
                writer.WriteAttributeString("type", (abObj.type.ToString()).Split('(')[0]);
                writer.WriteAttributeString("material", abObj.material.ToString());
                writer.WriteAttributeString("x", abObj.x.ToString("0." + new string('#', precisionPoints)));
                writer.WriteAttributeString("y", abObj.y.ToString("0." + new string('#', precisionPoints)));
                writer.WriteAttributeString("rotation", abObj.rotation.ToString("0." + new string('#', precisionPoints)));
                writer.WriteEndElement();
            }

            foreach (OBjData abObj in level.pigs)
            {
                writer.WriteStartElement("Pig");
                writer.WriteAttributeString("type", (abObj.type.ToString()).Split('(')[0]);
                writer.WriteAttributeString("x", abObj.x.ToString("0." + new string('#', precisionPoints)));
                writer.WriteAttributeString("y", abObj.y.ToString("0." + new string('#', precisionPoints)));
                writer.WriteAttributeString("rotation", abObj.rotation.ToString("0." + new string('#', precisionPoints)));
                writer.WriteEndElement();
            }

            foreach (OBjData abObj in level.tnts)
            {
                Debug.Log("TNT found in writing");
                writer.WriteStartElement("TNT");
                writer.WriteAttributeString("type", (abObj.type.ToString()).Split('(')[0]);
                writer.WriteAttributeString("x", abObj.x.ToString("0." + new string('#', precisionPoints)));
                writer.WriteAttributeString("y", abObj.y.ToString("0." + new string('#', precisionPoints)));
                writer.WriteAttributeString("rotation", abObj.rotation.ToString("0." + new string('#', precisionPoints)));
                writer.WriteEndElement();
            }

            foreach (PlatData abObj in level.platforms)
            {
                writer.WriteStartElement("Platform");
                writer.WriteAttributeString("type", (abObj.type.ToString()).Split('(')[0]);
                writer.WriteAttributeString("x", abObj.x.ToString("0." + new string('#', precisionPoints)));
                writer.WriteAttributeString("y", abObj.y.ToString("0." + new string('#', precisionPoints)));
                writer.WriteAttributeString("rotation", abObj.rotation.ToString("0." + new string('#', precisionPoints)));
                writer.WriteAttributeString("scaleX", abObj.scaleX.ToString("0." + new string('#', precisionPoints)));
                writer.WriteAttributeString("scaleY", abObj.scaleY.ToString("0." + new string('#', precisionPoints)));
                writer.WriteEndElement();
            }
        }

        StreamWriter streamWriter = new StreamWriter(path);
        streamWriter.WriteLine(output.ToString());
        streamWriter.Close();
    }

    public static Dictionary<string, GameObject> LoadABResource(string path) {

		// Load block templates and cast them to game objects
		UnityEngine.Object[] objs = Resources.LoadAll(path);

		Dictionary<string, GameObject> resources = new Dictionary<string, GameObject>();

		for (int i = 0; i < objs.Length; i++) {

			GameObject abGameObject = (GameObject)objs[i];
			resources [abGameObject.name] = abGameObject;
		}

		return resources;
	}

	public void SaveLevelOnScene() {

//		Transform blocksInScene = GameWorld.Instance.BlocksInScene();
//
//		List<GameObject> objsInScene = GameWorld.Instance.BlocksInScene();

//		ABLevel level = ABLevelGenerator.GameObjectsToABLevel(objsInScene.ToArray());
//		SaveXmlLevel(level);
	}
}