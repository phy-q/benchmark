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

﻿using UnityEditor;
using UnityEngine;
using System.Collections;

[CustomEditor(typeof(ABBlock))]
public class BlockEditor : Editor {

	// Update is called once per frame
	public override void OnInspectorGUI() {

		DrawDefaultInspector ();

		ABBlock block = (ABBlock)target;
		UpdateBlockMaterial (block, block._material);
	}

    public static void UpdateBlockMaterial(ABBlock block, MATERIALS material) {

        block._material = material;


        if (material == MATERIALS.wood) {
            block.GetComponent<SpriteRenderer>().sprite = block.GetComponent<ABBlock>()._woodSprites[0];
            block.GetComponent<Rigidbody2D>().sharedMaterial = ABWorldAssets.WOOD_MATERIAL;
            block.GetComponent<Collider2D>().sharedMaterial = ABWorldAssets.WOOD_MATERIAL;

        } else if (material == MATERIALS.stone) {
            block.GetComponent<SpriteRenderer>().sprite = block.GetComponent<ABBlock>()._stoneSprites[0];
            block.GetComponent<Rigidbody2D>().sharedMaterial = ABWorldAssets.STONE_MATERIAL;
            block.GetComponent<Collider2D>().sharedMaterial = ABWorldAssets.STONE_MATERIAL;

        }
        else if (material == MATERIALS.ice) {
            block.GetComponent<SpriteRenderer>().sprite = block.GetComponent<ABBlock>()._iceSprites[0];
            block.GetComponent<Rigidbody2D>().sharedMaterial = ABWorldAssets.ICE_MATERIAL;
            block.GetComponent<Collider2D>().sharedMaterial = ABWorldAssets.ICE_MATERIAL;

        }
    }
}
