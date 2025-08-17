const Joi = require('joi');

const promptSchema = Joi.object({
    vial_id: Joi.string().required(),
    prompt: Joi.string().required()
});

const taskSchema = Joi.object({
    vial_id: Joi.string().required(),
    task: Joi.string().required()
});

const configSchema = Joi.object({
    vial_id: Joi.string().required(),
    key: Joi.string().required(),
    value: Joi.string().required()
});

module.exports = {
    validatePrompt: (data) => promptSchema.validate(data),
    validateTask: (data) => taskSchema.validate(data),
    validateConfig: (data) => configSchema.validate(data)
};
