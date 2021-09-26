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
using System.IO;
using System.Collections.Generic;
public class LevelList : ABSingleton<LevelList> {

	private ABLevel[]   _levels;

	public int CurrentIndex;
	public int NextIndex;

	public ABLevel GetCurrentLevel() { 

		if (_levels == null)
			return null;

		if(CurrentIndex > _levels.Length - 1)
			return null;

		return _levels [CurrentIndex]; 
	}

	public void LoadLevelsFromSource(string[] levelSource, bool shuffle = false) {

		CurrentIndex = 0;

		_levels = new ABLevel[levelSource.Length];

		if(shuffle)
			ABArrayUtils.Shuffle(levelSource);

		for(int i = 0; i < levelSource.Length; i++)
			_levels[i] = LevelLoader.LoadXmlLevel(levelSource[i]);
	}

	public void RefreshLevelsFromSource(string[] levelSource, bool shuffle = false) {

		_levels = new ABLevel[levelSource.Length];

		if(shuffle)
			ABArrayUtils.Shuffle(levelSource);

		for(int i = 0; i < levelSource.Length; i++)
			_levels[i] = LevelLoader.LoadXmlLevel(levelSource[i]);
	}

	// Use this for initialization
	public ABLevel NextLevel() {

		if(CurrentIndex == _levels.Length - 1)
			return null;

		ABLevel level = _levels [CurrentIndex];
		CurrentIndex++;

		return level;
	}

	// Use this for initialization
	public ABLevel SetLevel(int index) {

		if(index < 0 || index >= _levels.Length)
			return null;

		CurrentIndex = index;
		ABLevel level = _levels [CurrentIndex];

		return level;
	}

    public int GetNumberOfLevels()
    {
		// update the level list if the Levels directory has been changed 
		//int levelCounter = ABLevelUpdate.RefreshLevelList();
		int levelCounter;
		levelCounter = LoadLevelSchema.Instance.currentXmlFiles.Length;
		
//		print("Total Number of Levels(resourcesXml.Length): " + levelCounter);

		return (levelCounter);
    }
}
