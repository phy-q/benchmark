using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;
using System.IO;
using System.Xml;
using System.Text;
using UnityEditor;

public class SaveLevel
{

    // [MenuItem("Novelty Generator/Save Level File")]
    public static void SaveXmlLevel()
    {

        Debug.Log("Writing the level file");


        ABLevel level = EncodeLevel();
        string path = "level.xml";

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
}
