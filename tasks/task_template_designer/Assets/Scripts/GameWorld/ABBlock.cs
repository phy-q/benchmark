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

public class ABBlock : ABGameObject
{

    public MATERIALS _material;

    public int _points;

    public Sprite[] _woodSprites;
    public Sprite[] _stoneSprites;
    public Sprite[] _iceSprites;

    /// <summary>
    /// Max hp - wood
    /// </summary>
    public float _woodLife;
    /// <summary>
    /// Max hp - stone
    /// </summary>
    public float _stoneLife;
    /// <summary>
    /// Max hp - ice
    /// </summary>
    public float _iceLife;

    protected override void Awake()
    {

        base.Awake();
        SetMaterial(_material);
    }

    public override void Die(bool withEffect = true)
    {
        if (!ABGameWorld.Instance._isSimulation)
            ScoreHud.Instance.SpawnScorePoint(_points, transform.position);

        base.Die();
    }

    public void SetMaterial(MATERIALS material)
    {

        _material = material;

        switch (material)
        {

            case MATERIALS.wood:
                _sprites = _woodSprites;
                _spriteRenderer.sprite = _sprites[0];
                _destroyEffect._particleSprites = ABWorldAssets.WOOD_DESTRUCTION_EFFECT;
                _collider.sharedMaterial = ABWorldAssets.WOOD_MATERIAL;
                getRigidBody().sharedMaterial = ABWorldAssets.WOOD_MATERIAL;

                // Density
                _density = ABConstants.DENSITY_WOOD;
                // HP - defined for each shapes seperately
                _life = _woodLife;
                // Defense
                _defense = ABConstants.DEFENSE_WOOD;
                break;

            case MATERIALS.stone:
                _sprites = _stoneSprites;
                _spriteRenderer.sprite = _sprites[0];
                _destroyEffect._particleSprites = ABWorldAssets.STONE_DESTRUCTION_EFFECT;
                _collider.sharedMaterial = ABWorldAssets.STONE_MATERIAL;
                getRigidBody().sharedMaterial = ABWorldAssets.STONE_MATERIAL;

                // Density
                _density = ABConstants.DENSITY_STONE;
                // HP - defined for each shapes seperately
                _life = _stoneLife;
                // Defense
                _defense = ABConstants.DEFENSE_STONE;
                break;

            case MATERIALS.ice:
                _sprites = _iceSprites;
                _spriteRenderer.sprite = _sprites[0];
                _destroyEffect._particleSprites = ABWorldAssets.ICE_DESTRUCTION_EFFECT;
                _collider.sharedMaterial = ABWorldAssets.ICE_MATERIAL;
                getRigidBody().sharedMaterial = ABWorldAssets.ICE_MATERIAL;

                // Density
                _density = ABConstants.DENSITY_ICE;
                // HP - defined for each shapes seperately
                _life = _iceLife;
                // Defense
                _defense = ABConstants.DEFENSE_ICE;
                break;

            default:
                break;
        }
    }

    public void SetMaterial(MATERIALS material, string objMaterial)
    {

        _material = material;

        switch (material)
        {

            case MATERIALS.novelty:
                /*
				*leave some space for future develop
				*/
                _collider.sharedMaterial = (PhysicsMaterial2D)ABGameWorld.NOVELTIES.LoadAsset(objMaterial);

                break;

            default:
                Debug.Log("wrong material choice for novel objects");
                break;

        }


    }

    public override void OnCollisionEnter2D(Collision2D collision)
    {
        if (collision.gameObject.tag == "Bird")
        {

            ABBird bird = collision.gameObject.GetComponent<ABBird>();

            float birdDamageMultiplier = 1f;

            switch (_material)
            {

                case MATERIALS.wood:
                    birdDamageMultiplier = bird._woodDamageMultiplier;
                    break;

                case MATERIALS.stone:
                    birdDamageMultiplier = bird._stoneDamageMultiplier;
                    break;

                case MATERIALS.ice:
                    birdDamageMultiplier = bird._iceDamageMultiplier;
                    break;

                case MATERIALS.novelty:
                    birdDamageMultiplier = 20; // just for this release, we fixed the bird damage
                    break;
            }

            // relative speed
            float relativeSpeed = collision.relativeVelocity.magnitude;

            // damage = relativeSpeed * massAttacker * damageMultiplier - defense
            // In this situation, the attacker is always the bird
            float damage = Mathf.Max(relativeSpeed * bird.getRigidBody().mass * birdDamageMultiplier * ABConstants.DAMAGE_CONSTANT_MULTIPLIER - _defense, 0);

            DealDamage(damage);


            /*
            // relative speed
            float relativeSpeed = collision.relativeVelocity.magnitude;

            float totalMass = _rigidBody.mass + bird.getRigidBody().mass;

            // damage = relativeSpeed * mass * bird damage multiplier * multiplier - defense
            float damage = Mathf.Max(relativeSpeed * totalMass * birdDamageMultiplier * ABConstants.DAMAGE_CONSTANT_MULTIPLIER - _defense, 0);

            // this object takes damage
            DealDamage(damage);
            */

            /*
            // 2D Impulse force calculation
            float impulse = 0;
            foreach (ContactPoint2D point in collision.contacts)
            {
                impulse += point.normalImpulse;
            }

            // damage = impulse * bird damage multiplier * multiplier - defense
           float damage = Mathf.Max(impulse * birdDamageMultiplier * ABConstants.DAMAGE_CONSTANT_MULTIPLIER - _defense, 0);

            DealDamage(damage);
            */

            /*
            // kinetic energy loss amount
            float kineticLoss = 0;

            // calculate kinetic energy difference of this collider
            kineticLoss +=
                0.5f * _rigidBody.mass * (lastVelocity.magnitude - _rigidBody.velocity.magnitude);

            // calculate kinetic energy difference of incoming bird
            kineticLoss += 
                0.5f * bird.getRigidBody().mass * (bird.lastVelocity.magnitude - bird.getRigidBody().velocity.magnitude); 

            // damage =  kinetic energy loss * multiplier * bird damage multiplier - defense
            float damage = Mathf.Max(0, kineticLoss * ABConstants.DAMAGE_CONSTANT_MULTIPLIER * birdDamageMultiplier - _defense);

            // this object takes damage
            DealDamage(damage);
            */

            // Debug.Log($"{gameObject} took {damage} damage from bird {bird.gameObject}.");

            // give points if it only exeeds 10 and block is not destroyed
            int points = (int)Mathf.Round(damage) * 10;
            if ((points > 10) & ((getCurrentLife() - damage) > 0f))
            {
                ScoreHud.Instance.SpawnScorePoint(points, transform.position);
            }
        }
        else
        {
            base.OnCollisionEnter2D(collision);
        }
    }
}
