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
using System.Collections;
using System.Collections.Generic;

public class OBjData {

	public string type;
	public float  rotation;
	public float  x, y;

	public OBjData () { }

	public OBjData(string type, float rotation, float x, float y) {

		this.type = type;
		this.rotation = rotation;
		this.x = x;
		this.y = y;
	}
}

public class BirdData {

	public string type;

	public BirdData(string type) {

		this.type = type;
	}
}

public class BlockData: OBjData {

	public string material;

	public BlockData(string type, float rotation, float x, float y, string material) {

		this.type = type;
		this.rotation = rotation;
		this.x = x;
		this.y = y;
		this.material = material;
	}
}

public class PlatData : OBjData {

	public float scaleX;
	public float scaleY;

	public PlatData () { }

	public PlatData(string type, float rotation, float x, float y, float scaleX, float scaleY) {

		this.type = type;
		this.rotation = rotation;
		this.x = x;
		this.y = y;
		this.scaleX = scaleX;
		this.scaleY = scaleY;
	}
}

public struct CameraData {

	public float minWidth;
	public float maxWidth;
	public float x, y;

	public CameraData(float minWidth, float maxWidth, float x, float y) {

		this.minWidth = minWidth;
		this.maxWidth = maxWidth;
		this.x = x;
		this.y = y;
	}
}

public struct SlingData {

	public float x, y;

	public SlingData(float x, float y) {

		this.x = x;
		this.y = y;
	}
}

//Adding the score 
public struct ScoreData
{
    public float highScore;

    public ScoreData(float highScore)
    {
        this.highScore = highScore;
    }
}

public class ABLevel 
{
	public int width;

	public string assetBundleFilePath;

	public CameraData camera;
	public SlingData slingshot;
    //Adding the score
    public ScoreData score;

	public List<OBjData>   pigs;
	public List<OBjData>   tnts;
	public List<BirdData>  birds;
	public List<BlockData> blocks;
	public List<PlatData>  platforms;

	public static readonly int BIRDS_MAX_AMOUNT = 5;

	public ABLevel() {

		width = 1;

		pigs      = new List<OBjData>();
		tnts      = new List<OBjData>();
		blocks    = new List<BlockData>();
		birds     = new List<BirdData>();
		platforms = new List<PlatData>();
		assetBundleFilePath = ABConstants.DEFAULT_ASSET_BUNDLE_FILE;
	}
    
}